/*
Original Code From:
Copyright (C) 2006 Pedro Felzenszwalb
Modifications (may have been made) Copyright (C) 2011, 2012
  Chenliang Xu, Jason Corso.

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

/* Implements the edge data structure. */

#ifndef EDGES_S_H
#define EDGES_S_H

#include <vector>
#include "image.h"
#include "disjoint-set.h"
#include "histogram.h"
#include "edges.h"

using namespace std;

__host__ __device__
float sqrt3(const float x)  
{
  union
  {
    int i;
    float x;
  } u;

  u.x = x;
  u.i = (1<<29) + (u.i >> 1) - (1<<22); 
  return u.x;
}

__host__ __device__
float inline square3(float n)
{
	return n*n;	
}

/* fill pixel level edges */
__host__ __device__
void generate_edge_s(edge *e, /*image<float> *r_v, image<float> *g_v,
		image<float> *b_v, image<float> *r_u, image<float> *g_u,
		image<float> *b_u,*/float *r, float *g, float *b,
		int x_v, int y_v, int z_v, int x_u, int y_u,
		int z_u, int width, int height, int offset) {
//	width = r_v->width();
//	height = r_v->height();
//	printf("x_v is %d and y_v %d, x_u %d and y_u %d.\n", x_v, y_v, x_u, y_u); 
//	printf("image data is %f.\n", r_v->imRef_s(r_v, x_v, y_v));
	e->a = y_v * width + x_v + z_v * (width * height);
	e->b = y_u * width + x_u + z_u * (width * height);
/*	e->w = sqrt3(
			square3(r_v->imRef_s(r_v, x_v, y_v) - r_u->imRef_s(r_u, x_u, y_u))
					+ square3(g_v->imRef_s(g_v, x_v, y_v) - g_u->imRef_s(g_u, x_u, y_u))
					+ square3(b_v->imRef_s(b_v, x_v, y_v) - b_u->imRef_s(b_u, x_u, y_u)));*/
	int index1 = ((z_v + offset) * height + y_v) * width + x_v;
	int index2 = ((z_u + offset) * height + y_u) * width + x_u;
	e->w = sqrt3(square3(r[index1] - r[index2]) + square3(g[index1] - g[index2]) + square3(b[index1] - b[index2]));

}


/* initialize pixel level edges */
__host__ __device__
void initialize_edges_s(edge *edges, int num_frame, int width, int height,
		/*image<float> *smooth_r[], image<float> *smooth_g[],
		image<float> *smooth_b[],*/ float *smooth_r, float *smooth_g, float *smooth_b, int case_num) {
        int offset = case_num * num_frame;
        
	int num_edges = 0;
	for (int z = 0; z < num_frame; z++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				// in the same plane
				if (x < width - 1) {
					generate_edge_s(&edges[num_edges], smooth_r, smooth_g, smooth_b, 
							x + 1, y, z, x, y, z, width, height, offset);
					num_edges++;
				}
				if (y < height - 1) {
					generate_edge_s(&edges[num_edges], smooth_r, smooth_g,
							smooth_b, x, y + 1, z, x, y, z, width, height, offset);
					num_edges++;
				}
				if ((x < width - 1) && (y < height - 1)) {
					generate_edge_s(&edges[num_edges], smooth_r, smooth_g,
							smooth_b, x + 1, y + 1, z, x, y, z, width, height, offset);
					num_edges++;
				}
				if ((x < width - 1) && (y > 0)) {
					generate_edge_s(&edges[num_edges], smooth_r, smooth_g,
							smooth_b, x + 1, y - 1, z, x, y, z, width, height, offset);
					num_edges++;
				}

				// to the previous plane
				if (z > 0) {

					generate_edge_s(&edges[num_edges], smooth_r,
					 		smooth_g, smooth_b, x, y, z - 1, x, y, z, width, height, offset);
					num_edges++;

					if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
						// additional 8 edges
						// x - 1, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x - 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
					} else if (x == 0 && y > 0 && y < height - 1) {
						// additional 5 edges
						// x, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
					} else if (x == width - 1 && y > 0 && y < height - 1) {
						// additional 5 edges
						// x - 1, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					} else if (y == 0 && x > 0 && x < width - 1) {
						// additional 5 edges
						// x - 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
					} else if (y == height - 1 && x > 0 && x < width - 1) {
						// additional 5 edges
						// x - 1, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x - 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					} else if (x == 0 && y == 0) {
						// additional 3 edges
						// x + 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
					} else if (x == 0 && y == height - 1) {
						// additional 3 edges
						// x, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					} else if (x == width - 1 && y == 0) {
						// additional 3 edges
						// x - 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					} else if (x == width - 1 && y == height - 1) {
						// additional 3 edges
						// x - 1, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y - 1
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y
						generate_edge_s(&edges[num_edges], smooth_r,
								smooth_g, smooth_b, x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					}

				}
			}
		}
	}

//	printf("num_edges = %d\n", num_edges);
}

#endif /* EDGES_S_H */
