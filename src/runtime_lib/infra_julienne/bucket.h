#pragma once

#include <limits>
#include <tuple>

#include "dyn_arr.h"
#include "maybe.h"
#include "parallel.h"
#include "vertexSubset.h"

#define CACHE_LINE_S 64

using namespace std;

typedef uintE bucket_id;
typedef uintE bucket_dest;

// Defines the order in which the buckets are accessed.
enum bucket_order {
  decreasing,
  increasing
};

// Defines the order in which priorities are updated.
enum priority_order {
   strictly_decreasing,
   strictly_increasing
};

struct bucket {
  size_t id;
  size_t num_filtered;
  vertexSubset identifiers;
  bucket(size_t _id, vertexSubset _identifiers) :
    id(_id), identifiers(_identifiers) { }
};

template <class D>
struct buckets {
  public:
    using id_dyn_arr = dyn_arr<uintE>;

    const uintE null_bkt = std::numeric_limits<D>::max();
    int delta_ = 1;

    // Create a bucketing structure.
    //   n : the number of identifiers
    //   d : map from identifier -> bucket
    //   bkt_order : the order to iterate over the buckets
    //   pri_order : the order in which priorities are updated
    //   total_buckets: the total buckets to materialize
    //
    //   For an identifier i:
    //   d[i] is the bucket currently containing i
    //   d[i] = UINT_E_MAX if i is not in any bucket
    buckets(size_t _n,
            D* _d,
            bucket_order _bkt_order,
            priority_order _pri_order,
            size_t _total_buckets, int delta=1) :
        n(_n), d(_d), bkt_order(_bkt_order), pri_order(_pri_order),
        open_buckets(_total_buckets-1), total_buckets(_total_buckets),
        cur_bkt(0), max_bkt(_total_buckets), num_elms(0), delta_(delta) {
      // Initialize array consisting of the materialized buckets.
      bkts = pbbso::new_array<id_dyn_arr>(total_buckets);

      // Set the current range being processed based on the order.
      if (bkt_order == increasing) {
//        auto imap = make_in_imap<uintE>(n, [&] (size_t i) { return d[i]; });
        auto imap = make_in_imap<uintE>(n, [&] (size_t i) { return (d[i] == null_bkt) ? null_bkt : d[i]/delta_; });
        auto min = [] (uintE x, uintE y) { return std::min(x, y); };
        size_t min_b = pbbso::reduce(imap, min);
        cur_range = min_b / open_buckets;
      } else if (bkt_order == decreasing) {
        auto imap = make_in_imap<uintE>(n, [&] (size_t i) {
            return (d[i] == null_bkt) ? 0 : d[i]/delta; });
        auto max = [] (uintE x, uintE y) { return std::max(x,y); };
        size_t max_b = pbbso::reduce(imap, max);
        cur_range = (max_b + open_buckets) / open_buckets;
      } else {
        cout << "Unknown order: " << bkt_order
             << ". Must be one of {increasing, decreasing}" << endl;
        abort();
      }

      // Update buckets with all (id, bucket) pairs. Identifiers with bkt =
      // null_bkt are ignored by update_buckets.
      auto get_id_and_bkt = [&] (uintE i) -> Maybe<tuple<uintE, uintE> > {
          //updated with delta
        uintE bkt = (d[i] == null_bkt) ? null_bkt : d[i]/delta_;
        if (bkt != null_bkt) {
          bkt = to_range(bkt);
        }
        return Maybe<tuple<uintE, uintE> >(make_tuple(i, bkt));
      };
      update_buckets(get_id_and_bkt, n);
    }

    // Returns the next non-empty bucket from the bucket structure. The return
    // value's bkt_id is null_bkt when no further buckets remain.
    inline bucket next_bucket() {
      while (!curBucketNonEmpty() && num_elms > 0) {
        _next_bucket();
      }
      if (num_elms == 0) {
        size_t bkt_num = null_bkt; // no buckets remain
        vertexSubset vs(n);
        return bucket(bkt_num, vs);
      }
      return get_cur_bucket();
    }

    // Computes a bucket_dest for an identifier moving to bucket_id next.
    inline bucket_dest get_bucket_no_overflow_insertion(const bucket_id& next) const {
      uintE nb = to_range(next);
      // Note that the interface currently only implements strictly_decreasing 
      // priority, which is why the code below does not check pri_order.
      if (bkt_order == increasing) {
        // case for strictly_decreasing priorities, assuming elements start out
        // in the structure.
        // nb != open_buckets prevent it from returning the overflow bucket (open_buckets) as a target
        if (nb != null_bkt && nb != open_buckets) {
          return nb;
        } // case for strictly_increasing elided
      } else { // bkt_order == decreasing
        if (nb != null_bkt) {
        // strictly_decreasing priorities, assuming elements start out in the structure.
          return nb;
        }
      }
      return null_bkt;
    }

    // Computes a bucket_dest for an identifier moving to bucket_id next.
    inline bucket_dest get_bucket_with_overflow_insertion(const bucket_id& next) const {
        uintE nb = to_range(next);
        // Note that the interface currently only implements strictly_decreasing
        // priority, which is why the code below does not check pri_order.
        if (bkt_order == increasing) {
            // case for strictly_decreasing priorities, assuming elements start out
            // in the structure.
            if (nb != null_bkt ) {
                return nb;
            } // case for strictly_increasing elided
        } else { // bkt_order == decreasing
            if (nb != null_bkt) {
                // strictly_decreasing priorities, assuming elements start out in the structure.
                return nb;
            }
        }
        return null_bkt;
    }


    // Updates k identifiers in the bucket structure. The i'th identifier and
    // its bucket_dest are given by F(i).
    template <class F>
    inline size_t update_buckets(F f, size_t k) {
       size_t num_blocks = k / 4096;
       int num_threads = getWorkers();
       if (k < 4096 || num_threads == 1) {
         return update_buckets_seq(f, k);
       }

       size_t ne_before = num_elms;

       size_t block_bits = pbbso::log2_up(num_blocks);
       num_blocks = 1 << block_bits;
       size_t block_size = (k + num_blocks - 1) / num_blocks;

       uintE* hists = pbbso::new_array_no_init<uintE>((num_blocks+1) * total_buckets * CACHE_LINE_S);
       uintE* outs = pbbso::new_array_no_init<uintE>((num_blocks+1) * total_buckets);

       // 1. Compute per-block histograms
       parallel_for_1(size_t i=0; i<num_blocks; i++) {
         size_t s = i * block_size;
         size_t e = min(s + block_size, k);
         uintE* hist = &(hists[i*total_buckets]);

         for (size_t j=0; j<total_buckets; j++) { hist[j] = 0; }
         for (size_t j=s; j<e; j++) {
           auto m = f(j);
           bucket_dest b = std::get<1>(m.t);
           if (m.exists && b != null_bkt) {
             hist[b]++;
           }
         }
       }

      // 2. Aggregate histograms into a single histogram.
      auto get = [&] (size_t i) {
        size_t col = i % num_blocks;
        size_t row = i / num_blocks;
        return hists[col*total_buckets + row];
      };

      auto in_map = make_in_imap<uintE>(num_blocks*total_buckets, get);
      auto out_map = array_imap<uintE>(outs, num_blocks*total_buckets);

      size_t sum = pbbso::scan_add(in_map, out_map);
      outs[num_blocks*total_buckets] = sum;

      // 3. Resize buckets based on the summed histogram.
      for (size_t i=0; i<total_buckets; i++) {
        size_t num_inc = outs[(i+1)*num_blocks] - outs[i*num_blocks];
        bkts[i].resize(num_inc);
        num_elms += num_inc;
      }

      // 4. Compute the starting offsets for each block.
      parallel_for(size_t i=0; i<total_buckets; i++) {
        size_t start = outs[i*num_blocks];
        for (size_t j=0; j<num_blocks; j++) {
          hists[(i*num_blocks + j)*CACHE_LINE_S] = outs[i*num_blocks + j] - start;
        }
      }

      // 5. Iterate over blocks again. Insert (id, bkt) into bkt[hists[bkt]]
      // and increment hists[bkt].
      parallel_for_1 (size_t i=0; i<num_blocks; i++) {
         size_t s = i * block_size;
         size_t e = min(s + block_size, k);
         // our buckets are now spread out, across outs
         for (size_t j=s; j<e; j++) {
           auto m = f(j);
           uintE v = std::get<0>(m.t);
           bucket_dest b = std::get<1>(m.t);
           if (m.exists && b != null_bkt) {
             size_t ind = hists[(b*num_blocks + i)*CACHE_LINE_S];
             bkts[b].insert(v, ind);
             hists[(b*num_blocks + i)*CACHE_LINE_S]++;
           }
         }
      }

      // 6. Finally, update the size of each bucket.
      for (size_t i=0; i<total_buckets; i++) {
        size_t num_inc = outs[(i+1)*num_blocks] - outs[i*num_blocks];
        size_t& m = bkts[i].size;
        m += num_inc;
      }

      free(hists); free(outs);
      return num_elms - ne_before;
    }

  private:
    const bucket_order bkt_order;
    const priority_order pri_order;
    id_dyn_arr* bkts;
    size_t cur_bkt;
    size_t max_bkt;
    size_t cur_range;
    D* d;
    size_t n; // total number of identifiers in the system
    size_t num_elms;
    size_t open_buckets;
    size_t total_buckets;

    template <class F>
    inline size_t update_buckets_seq(F& f, size_t n) {
      size_t ne_before = num_elms;
      for (size_t i=0; i<n; i++) {
        auto m = f(i);
        bucket_dest bkt = std::get<1>(m.t);
        if (m.exists && bkt != null_bkt) {
          bkts[bkt].resize(1);
          insert_in_bucket(bkt, std::get<0>(m.t));
          num_elms++;
        }
      }
      return num_elms - ne_before;
    }

    inline void insert_in_bucket(size_t b, intT val) {
      uintE* dst = bkts[b].A;
      intT size = bkts[b].size;
      dst[size] = val;
      bkts[b].size += 1;
    }

    inline bool curBucketNonEmpty() {
      return bkts[cur_bkt].size > 0;
    }

    inline void unpack() {
      size_t m = bkts[open_buckets].size;
      auto tmp = array_imap<uintE>(m);
      uintE* A = bkts[open_buckets].A;
      parallel_for(size_t i=0; i<m; i++) {
        tmp[i] = A[i];
      }
      if (bkt_order == increasing) {
        cur_range++; // increment range
      } else {
        cur_range--;
      }
      bkts[open_buckets].size = 0; // reset size

      auto g = [&] (uintE i) -> Maybe<tuple<uintE, uintE> > {
        uintE v = tmp[i];
        //uintE bkt = to_range(d[v]);
        D priority = (d[v] == null_bkt)? null_bkt : d[v]/delta_;
        uintE bkt = to_range(priority);
          return Maybe<tuple<uintE, uintE> >(make_tuple(v, bkt));
      };

      if (m != num_elms) {
        cout << "m = " << m << " num_elms = " << num_elms << endl;
        cur_bkt = 0;
        cout << "curBkt = " << get_cur_bucket_num() << endl;
        cout << "mismatch" << endl;
        for (size_t i=0; i<total_buckets; i++) {
          cout << bkts[i].size << endl;
          if (bkts[i].size > 0) {
            for (size_t j=0; j<bkts[i].size; j++) {
              cout << bkts[i].A[j] << endl;
              cout << "deg = " << d[bkts[i].A[j]] << endl;
              cout << "bkt = " << ((cur_range+1)*(open_buckets) - i - 1) << endl;
            }
          }
        }
        exit(0);
      }
      size_t updated = update_buckets(g, m);
      size_t num_in_range = updated - bkts[open_buckets].size;
      num_elms -= m;
    }

    inline void _next_bucket() {
      cur_bkt++;
      if (cur_bkt == open_buckets) {
        unpack();
        cur_bkt = 0;
      }
    }

    // increasing: [cur_range*open_buckets, (cur_range+1)*open_buckets)
    // decreasing: [(cur_range-1)*open_buckets, cur_range*open_buckets)
    inline bucket_id to_range(uintE bkt) const {
      if (bkt_order == increasing) {
        if (bkt < cur_range*open_buckets) { // this can happen because of the lazy bucketing
          return null_bkt;
        }
        return (bkt < (cur_range+1)*open_buckets) ? (bkt % open_buckets) : open_buckets;
      } else {
        if (bkt >= (cur_range)*open_buckets) {
          return null_bkt;
        }
        return (bkt >= (cur_range-1)*open_buckets) ? ((open_buckets - (bkt % open_buckets)) - 1) : open_buckets;
      }
    }

    size_t get_cur_bucket_num() const {
      if (bkt_order == increasing) {
        return cur_range*open_buckets + cur_bkt;
      } else {
        return (cur_range)*(open_buckets) - cur_bkt - 1;
      }
    }

    inline bucket get_cur_bucket() {
      id_dyn_arr bkt = bkts[cur_bkt];
      size_t size = bkt.size;
      num_elms -= size;
      uintE* out = newA(uintE, size);
      size_t cur_bkt_num = get_cur_bucket_num();
      auto p = [&] (size_t i) { return ((d[i] == null_bkt)? null_bkt : d[i]/delta_) == cur_bkt_num; };
      size_t m = pbbso::filterf(bkt.A, out, size, p);
      bkts[cur_bkt].size = 0;
      if (m == 0) {
        free(out);
        return next_bucket();
      }
      vertexSubset vs(n, m, out);
      auto ret = bucket(cur_bkt_num, vs);
      ret.num_filtered = size;
      return ret;
    }
};

template <class D>
buckets<D> make_buckets(size_t n, D* d, bucket_order bkt_order, priority_order pri_order, size_t total_buckets=128) {
  return buckets<D>(n, d, bkt_order, pri_order, total_buckets);
}
