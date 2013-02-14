/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SEGMENT_GRAPH_S
#define SEGMENT_GRAPH_S

#include <algorithm>
#include <cmath>
#include "disjoint-set-s.h"
#include <sys/time.h>
#include <iostream>
#include "edges.h"
// threshold function
#define THRESHOLD(size, c) (c/size)
/*
typedef struct {
  float w;
  int a, b;
} edge_s;
*/
__host__ __device__
bool operator<(const edge &a, const edge &b) {
  return a.w < b.w;
}

__host__ __device__
bool operator>(const edge &a, const edge &b) {
  return a.w > b.w;
}

template<class T>
__host__ __device__ void quicksort(T a[], const int& leftarg, const int& rightarg)
{
  if (leftarg < rightarg) {

    T pivotvalue = a[leftarg];
    int left = leftarg - 1;
    int right = rightarg + 1;

  for(;;) {

    while (a[--right] > pivotvalue);
    while (a[++left] < pivotvalue);

    if (left >= right) break;

    T temp = a[right];
    a[right] = a[left];
    a[left] = temp;
  }

  int pivot = right;
  quicksort(a, leftarg, pivot);
  quicksort(a, pivot + 1, rightarg);
  }
}

__host__ __device__ int back_push(edge *edges_remain[], edge *pedge, int cur_it) {
  edges_remain[cur_it] = pedge;
  ++cur_it;
  return cur_it;
}

template<class TYPE>
void algo_mergesort(size_t n, TYPE array[], TYPE *temp = 0)
{
	TYPE *a2[2], *a, *b;
	int curr, shift;

	a2[0] = array;
	a2[1] = temp? temp : new TYPE[n];
	for (curr = 0, shift = 0; (1ul<<shift) < n; ++shift) {
		a = a2[curr]; b = a2[1-curr];
		if (shift == 0) {
			TYPE *p = b, *i, *eb = a + n;
			for (i = a; i < eb; i += 2) {
				if (i == eb - 1) *p++ = *i;
				else {
					if (*(i+1) < *i) {
						*p++ = *(i+1); *p++ = *i;
					} else {
						*p++ = *i; *p++ = *(i+1);
					}
				}
			}
		} else {
			size_t i, step = 1ul<<shift;
			for (i = 0; i < n; i += step<<1) {
				TYPE *p, *j, *k, *ea, *eb;
				if (n < i + step) {
					ea = a + n; eb = a;
				} else {
					ea = a + i + step;
					eb = a + (n < i + (step<<1)? n : i + (step<<1));
				}
				j = a + i; k = a + i + step; p = b + i;
				while (j < ea && k < eb) {
					if (*j < *k) *p++ = *j++;
					else *p++ = *k++;
				}
				while (j < ea) *p++ = *j++;
				while (k < eb) *p++ = *k++;
			}
		}
		curr = 1 - curr;
	}
	if (curr == 1) {
		TYPE *p = a2[0], *i = a2[1], *eb = array + n;
		for (; p < eb; ++i) *p++ = *i;
	}
	if (temp == 0) delete[] a2[1];
}

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
__host__ __device__
/*universe_s **/void segment_graph_s(int num_vertices, int num_edges, edge *edges, 
			float c, /*vector<edge>* edges_remain*/ edge *edges_remain[],
                        universe_s *u, float *threshold) { 
  // new vector containing remain edges
//  edges_remain->clear();
  edges_remain = NULL;   
  int cur_it = 0; // current available iterator

  // sort edges by weight
  quicksort<edge>(edges, 0, num_edges - 1);

  // make a disjoint-set forest
//  universe_s *u = new universe_s(num_vertices);

  // init thresholds
//  float *threshold = new float[num_vertices];
  for (int i = 0; i < num_vertices; i++)
    threshold[i] = THRESHOLD(1,c);

  // for each edge, in non-decreasing weight order...
  for (int i = 0; i < num_edges; i++) {
    edge *pedge = &edges[i];
    
    // components conected by this edge
    int a = u->find(pedge->a);
    int b = u->find(pedge->b);
    if (a != b) {
      if ((pedge->w <= threshold[a]) &&
	  (pedge->w <= threshold[b])) {
	if (u->join(a, b, pedge->w) == 1)
          cur_it = back_push(edges_remain, pedge, cur_it);
	a = u->find(a);
	threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
      }
      else cur_it = back_push(edges_remain, pedge, cur_it);
    }
  }

  // free up
//  delete threshold;
//  return u;
}

#endif
