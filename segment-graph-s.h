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
universe_s *segment_graph_s(int num_vertices, int num_edges, edge *edges, 
			float c, vector<edge>* edges_remain) { 
  // new vector containing remain edges
  edges_remain->clear();
  // sort edges by weight
  std::sort(edges, edges + num_edges);

  // make a disjoint-set forest
  universe_s *u = new universe_s(num_vertices);

  // init thresholds
  float *threshold = new float[num_vertices];
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
          edges_remain->push_back(*pedge);
	a = u->find(a);
	threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
      }
      else edges_remain->push_back(*pedge);
    }
  }

  // free up
  delete threshold;
  return u;
}

#endif
