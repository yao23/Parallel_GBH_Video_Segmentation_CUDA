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
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "image.h"
#include "pnmfile.h"
//#include "segment-image-multi.h"
#include "disjoint-set.h"

#include <iostream> // from segment-image-multi.h
#include <fstream>
#include <vector>
#include <unistd.h>
#include <omp.h>
#include "edges.h"
#include "misc.h"
#include "filter.h"
#include "disjoint-set.h"
#include "segment-graph-multi.h"

#include <algorithm> // from segment-graph-multi.h
#include <cmath>
#include "disjoint-set-s.h"
#include "segment-graph-s.h"

#include "histogram.h"
//#include "edges-s.h"
#include <cuda.h>
#define num_cores 4 
#define num_smooth_s 12288000

using namespace std;

__constant__ float smooth_r1[num_smooth_s];
__constant__ float smooth_g1[num_smooth_s];
__constant__ float smooth_b1[num_smooth_s];

/* Save Output for oversegmentation*/
void generate_output_s(char *path, int num_frame, int width, int height,
                 universe_s *u, int num_vertices, int case_num) {

	int offset = case_num * num_frame; 
        char savepath[1024];
        image<rgb>** output = new image<rgb>*[num_frame];
        rgb* colors = new rgb[num_vertices];
        for (int i = 0; i < num_vertices; i++)
               colors[i] = random_rgb();

        // write out the ppm files.
        int k = 0;
        for (int i = 0; i < num_frame; i++) {
               snprintf(savepath, 1023, "%s/%02d/%05d.ppm", path, k, i + offset + 1);
               output[i] = new image<rgb>(width, height);
               for (int y = 0; y < height; y++) {
                      for (int x = 0; x < width; x++) {
                             int comp = u->find(y * width + x + i * (width * height));
                             imRef(output[i], x, y) = colors[comp];
                      }
               }
               savePPM(output[i], savepath);
        }

	#pragma omp parallel for 
        for (int i = 0; i < num_frame; i++)
               delete output[i];

        delete[] colors;
        delete[] output;
}
/*
__device__ void change4(int *x) { *x = 20121111; printf("In cuda1 change4, x is %d.----------------------------\n", *x);}
*/

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
		image<float> *b_u,float *r, float *g, float *b,*/
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
	e->w = sqrt3(square3(smooth_r1[index1] - smooth_r1[index2]) + square3(smooth_g1[index1] - smooth_g1[index2]) 
                             + square3(smooth_b1[index1] - smooth_b1[index2]));

}


/* initialize pixel level edges */
__host__ __device__
void initialize_edges_s(edge *edges, int num_frame, int width, int height,
		/*image<float> *smooth_r[], image<float> *smooth_g[],
		image<float> *smooth_b[],*/ /*float *smooth_r, float *smooth_g, float *smooth_b,*/ int case_num) {
        int offset = case_num * num_frame;
        
	int num_edges = 0;
	for (int z = 0; z < num_frame; z++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				// in the same plane
				if (x < width - 1) {
					generate_edge_s(&edges[num_edges],/* smooth_r1, smooth_g1, smooth_b1, */
							x + 1, y, z, x, y, z, width, height, offset);
					num_edges++;
				}
				if (y < height - 1) {
					generate_edge_s(&edges[num_edges],/* smooth_r1, smooth_g1,
							smooth_b1,*/ x, y + 1, z, x, y, z, width, height, offset);
					num_edges++;
				}
				if ((x < width - 1) && (y < height - 1)) {
					generate_edge_s(&edges[num_edges],/* smooth_r1, smooth_g1,
							smooth_b1,*/ x + 1, y + 1, z, x, y, z, width, height, offset);
					num_edges++;
				}
				if ((x < width - 1) && (y > 0)) {
					generate_edge_s(&edges[num_edges],/* smooth_r1, smooth_g1,
							smooth_b1,*/ x + 1, y - 1, z, x, y, z, width, height, offset);
					num_edges++;
				}

				// to the previous plane
				if (z > 0) {

					generate_edge_s(&edges[num_edges],/* smooth_r1,
					 		smooth_g1, smooth_b1,*/ x, y, z - 1, x, y, z, width, height, offset);
					num_edges++;

					if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
						// additional 8 edges
						// x - 1, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x - 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
					} else if (x == 0 && y > 0 && y < height - 1) {
						// additional 5 edges
						// x, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
					} else if (x == width - 1 && y > 0 && y < height - 1) {
						// additional 5 edges
						// x - 1, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					} else if (y == 0 && x > 0 && x < width - 1) {
						// additional 5 edges
						// x - 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
					} else if (y == height - 1 && x > 0 && x < width - 1) {
						// additional 5 edges
						// x - 1, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x - 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					} else if (x == 0 && y == 0) {
						// additional 3 edges
						// x + 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
					} else if (x == 0 && y == height - 1) {
						// additional 3 edges
						// x, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x + 1, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x + 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x + 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					} else if (x == width - 1 && y == 0) {
						// additional 3 edges
						// x - 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y + 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y + 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y + 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					} else if (x == width - 1 && y == height - 1) {
						// additional 3 edges
						// x - 1, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y - 1, z - 1,
								x, y, z, width, height, offset);
						num_edges++;
						// x, y - 1
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x, y - 1, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
						// x - 1, y
						generate_edge_s(&edges[num_edges],/* smooth_r1,
								smooth_g1, smooth_b1,*/ x - 1, y, z - 1, x, y,
								z, width, height, offset);
						num_edges++;
					}

				}
			}
		}
	}
}


// process edges initialization
__global__ void gb1(/*float *smooth_r, float *smooth_g, float *smooth_b,*/ int width, int height,  
                      edge *edges0, edge *edges1, edge *edges2, edge *edges3) {
  int case_num = blockIdx.x;
  int num_frame = blockDim.x;// * 20;
//      *x = 1111;
//  printf("In cuda1 change, x is %d.---------------------------------------------------------------\n", *x); 

  switch(case_num) {
    case 0: 
    {
      initialize_edges_s(edges0, num_frame, width, height,/* smooth_r1, smooth_g1, smooth_b1,*/ 0);
      //  printf("Finished edge initialization.\n");
    }
    break;
    case 1: 
    {
      initialize_edges_s(edges1, num_frame, width, height,/* smooth_r1, smooth_g1, smooth_b1,*/ 1);
      //  printf("Finished edge initialization.\n");
    }
    break;
    case 2: 
    {
      initialize_edges_s(edges2, num_frame, width, height,/* smooth_r1, smooth_g1, smooth_b1,*/ 2);
      //  printf("Finished edge initialization.\n");
    }
    break;
    case 3: 
    {
      initialize_edges_s(edges3, num_frame, width, height,/* smooth_r1, smooth_g1, smooth_b1,*/ 3);
      //  printf("Finished edge initialization.\n");
    }
    break;
    default: break;
  }
//  printf("Finished edges-initialization.\n");
}

// process segment-graph for units
__global__ void gb2(int width, int height, float c, edge *edges_remain0[], edge *edges_remain1[], 
		    edge *edges_remain2[], edge *edges_remain3[], universe_s *u0, universe_s *u1, 
		    universe_s *u2, universe_s *u3, int *er_num, edge *edges0, edge *edges1, edge *edges2, 
        	    edge *edges3) {
  // er_num is the array to record edge_remain element number
  int case_num = blockIdx.x;
  int num_frame = blockDim.x;// * 20;
//      *x = 1111;
//  printf("In cuda1 change, x is %d.---------------------------------------------------------------\n", *x); 
  // ----- node number
  int num_vertices = num_frame * width * height;
  // ----- edge number
  int num_edges_plane = (width - 1) * (height - 1) * 2 + width * (height - 1) + (width - 1) * height;
  int num_edges_layer = (width - 2) * (height - 2) * 9 + (width - 2) * 2 * 6 + (height - 2) * 2 * 6 + 4 * 4;
  int num_edges_s = num_edges_plane * num_frame + num_edges_layer * (num_frame - 1);
  switch(case_num) {
    case 0: 
    {
      er_num[0] = segment_graph_s(num_vertices, num_edges_s, edges0, c, edges_remain0, u0);//, x);
      //  printf("Finished unit graph segmentation.\n"); 
    }
    break;
    case 1: 
    {
      er_num[1] = segment_graph_s(num_vertices, num_edges_s, edges1, c, edges_remain1, u1);//, x);
      //  printf("Finished unit graph segmentation.\n"); 
    }
    break;
    case 2: 
    {
      er_num[2] = segment_graph_s(num_vertices, num_edges_s, edges2, c, edges_remain2, u2);//, x);
      //  printf("Finished unit graph segmentation.\n"); 
    }
    break;
    case 3: 
    {
      er_num[3] = segment_graph_s(num_vertices, num_edges_s, edges3, c, edges_remain3, u3);//, x);
      //  printf("Finished unit graph segmentation.\n");
    }
    break;
    default: break;
  }
//  printf("Finished segment-graph for units.\n");
}
/*
inline void check_cuda_errors(const char *filename, const int line_number)
{
#ifdef DEBUG
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
    exit(-1);
  }
#endif
}
*/
/* pixel level minimum spanning tree merge */
void segment_graph(universe *mess, vector<edge>* edges_remain, edge *edges, float c, int width, int height, int level,
                image<float> *smooth_r[], image<float> *smooth_g[], image<float> *smooth_b[], int num_frame, char *path) {
	// new vector containing remain edges
	edges_remain->clear();
	printf("Start segmenting graph in parallel.\n");
       
	int er_num[4] = {0}; // edegs_remain elements number
//        for (int i = 0; i < 4; ++i)
// 	  er_num[i] = i; 
        int *d_er_num = {0};

	// ----- edge number
        int num_edges_plane = (width - 1) * (height - 1) * 2 + width * (height - 1) + (width - 1) * height;
        int num_edges_layer = (width - 2) * (height - 2) * 9 + (width - 2) * 2 * 6 + (height - 2) * 2 * 6 + 4 * 4;
        int num_edges_s = num_edges_plane * num_frame + num_edges_layer * (num_frame - 1);
        // ----- node number
        int num_vertices = num_frame * width * height;
        int num_bytes = num_edges_s * sizeof(edge); // edge array size
	// smooth_r, smooth_g, smooth_b size
	int num_smooth = num_cores * num_vertices * sizeof(float);

	int block_size = num_frame; //ThreadsPerBlock
	int grid_size = num_cores; //BlocksPerGrid
	printf("grid_size is %d and block_size is %d.\n", grid_size, block_size);

	universe_s *u0 = new universe_s(num_vertices); universe_s *u1 = new universe_s(num_vertices); 
	universe_s *u2 = new universe_s(num_vertices); universe_s *u3 = new universe_s(num_vertices);
        int num_bytes_n = sizeof(u0);
	
	// copy edges from cpu to gpu
	edge *edges0 = new edge[num_edges_s]; edge *edges1 = new edge[num_edges_s]; 
	edge *edges2 = new edge[num_edges_s]; edge *edges3 = new edge[num_edges_s]; 
	edge *d_edges0 = NULL; edge *d_edges1 = NULL; edge *d_edges2 = NULL; edge *d_edges3 = NULL;
   	cudaMalloc((void**)&d_edges0, num_bytes); cudaMalloc((void**)&d_edges1, num_bytes);
   	cudaMalloc((void**)&d_edges2, num_bytes); cudaMalloc((void**)&d_edges3, num_bytes);

	// initialize edges and remained edges array	
	edge *edges_remain0 = new edge[num_edges_s]; edge **d_edges_remain0 = NULL; 
	edge *edges_remain1 = new edge[num_edges_s]; edge **d_edges_remain1 = NULL; 
	edge *edges_remain2 = new edge[num_edges_s]; edge **d_edges_remain2 = NULL; 
	edge *edges_remain3 = new edge[num_edges_s]; edge **d_edges_remain3 = NULL; 
	// cudaMalloc memory space for edges_remain element number counter
        cudaMalloc((void**)&d_er_num, 4*sizeof(int)); cudaMemcpy(d_er_num, er_num, 4*sizeof(int), cudaMemcpyHostToDevice);
	// cudaMalloc memory space for edge vectors 
        cudaMalloc((void**)&d_edges_remain0, num_bytes); cudaMemcpy(d_edges_remain0, edges_remain0, num_bytes, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_edges_remain1, num_bytes); cudaMemcpy(d_edges_remain1, edges_remain1, num_bytes, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_edges_remain2, num_bytes); cudaMemcpy(d_edges_remain2, edges_remain2, num_bytes, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_edges_remain3, num_bytes); cudaMemcpy(d_edges_remain3, edges_remain3, num_bytes, cudaMemcpyHostToDevice);

        // initialize node array 
        universe_s *d_u0 = new universe_s(num_vertices); universe_s *d_u1 = new universe_s(num_vertices); 
	universe_s *d_u2 = new universe_s(num_vertices); universe_s *d_u3 = new universe_s(num_vertices);

        // allocate memory space for node array 
//        cudaMalloc((void**)&d_u0, num_bytes_n); cudaMalloc((void**)&d_u1, num_bytes_n);
//        cudaMalloc((void**)&d_u2, num_bytes_n); cudaMalloc((void**)&d_u3, num_bytes_n);
        cudaMalloc((void**)&d_u0, sizeof(u0)); cudaMalloc((void**)&d_u1, sizeof(u1));
        cudaMalloc((void**)&d_u2, sizeof(u2)); cudaMalloc((void**)&d_u3, sizeof(u3));
       	cudaMemcpy(d_u0, u0, num_bytes_n, cudaMemcpyHostToDevice); cudaMemcpy(d_u1, u1, num_bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u2, u2, num_bytes_n, cudaMemcpyHostToDevice); cudaMemcpy(d_u3, u3, num_bytes_n, cudaMemcpyHostToDevice);

        // allocate gpu space for smooth_r, smooth_g, smooth_b 
	float* d_smooth_r; cudaMalloc((void**)&d_smooth_r, num_smooth);
	float* d_smooth_g; cudaMalloc((void**)&d_smooth_g, num_smooth);
	float* d_smooth_b; cudaMalloc((void**)&d_smooth_b, num_smooth);
	float* t_smooth_r = (float*)malloc(num_smooth);//new float[num_smooth];
	float* t_smooth_g = (float*)malloc(num_smooth);//new float[num_smooth];
	float* t_smooth_b = (float*)malloc(num_smooth);//new float[num_smooth];

	printf("begin to read image data.\n");	
	int tindex = 0;
	for (int i = 0; i < num_cores*num_frame; i++) {
                 for (int y = 0; y < height; y++) {
                         for (int x = 0; x < width; x++) {
                                 tindex = (i * height + y) * width + x; 
                                 t_smooth_r[tindex] = imRef(smooth_r[i], x, y);
                                 t_smooth_g[tindex] = imRef(smooth_g[i], x, y);
                                 t_smooth_b[tindex] = imRef(smooth_b[i], x, y);
                         }
                 }
        }
/*	for (int i = 0; i < num_cores*num_frame; i++) {
                 for (int y = 0; y < height; y++) {
                         for (int x = 0; x < width; x++) {
                                 tindex = (i * height + y) * width + x; 
                                 t_smooth_r[tindex] = imRef(smooth_r[i], x, y);
                                 t_smooth_g[tindex] = imRef(smooth_g[i], x, y);
                                 t_smooth_b[tindex] = imRef(smooth_b[i], x, y);
                         }
                 }
        }
*/	printf("begin to copy image data from cpu to gpu.\n");	
//        cudaMemcpy(d_smooth_r, t_smooth_r, num_smooth, cudaMemcpyHostToDevice);
//        cudaMemcpy(d_smooth_g, t_smooth_g, num_smooth, cudaMemcpyHostToDevice);
//        cudaMemcpy(d_smooth_b, t_smooth_b, num_smooth, cudaMemcpyHostToDevice);
	cudaError_t err0_1, err0_2, err0_3;
	err0_1 = cudaMemcpyToSymbol("smooth_r1", &t_smooth_r[0], num_smooth_s, size_t(0), cudaMemcpyHostToDevice);
	err0_2 = cudaMemcpyToSymbol("smooth_g1", &t_smooth_g[0], num_smooth_s, size_t(0), cudaMemcpyHostToDevice);
	err0_3 = cudaMemcpyToSymbol("smooth_b1", &t_smooth_b[0], num_smooth_s, size_t(0), cudaMemcpyHostToDevice);
	printf("Error0_1: %s\n", cudaGetErrorString(err0_1) );
	printf("Error0_2: %s\n", cudaGetErrorString(err0_2) );
	printf("Error0_3: %s\n", cudaGetErrorString(err0_3) );

	printf("Start segmenting graph in GPU.\n");

	int deviceId = 2;
	cudaThreadExit(); // clears all the runtime state for the current thread
	cudaSetDevice(deviceId); // explicit set the current device for the other calls

	cudaError_t err, err2, err3, err4;
	const char *err5;
	gb1<<<grid_size,block_size>>>(/*d_smooth_r, d_smooth_g, d_smooth_b,*/ width, height,  
                                      d_edges0, d_edges1, d_edges2, d_edges3);
	// transfter edges from gpu to cpu for next unit-segment-graph
        err2 = cudaMemcpy(edges0, d_edges0, num_bytes, cudaMemcpyDeviceToHost);
	printf("Error2: %s\n", cudaGetErrorString(err2) );
	cudaMemcpy(edges1, d_edges1, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(edges2, d_edges2, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(edges3, d_edges3, num_bytes, cudaMemcpyDeviceToHost);

        cudaMemcpy(d_edges0, edges0, num_bytes, cudaMemcpyHostToDevice); cudaMemcpy(d_edges1, edges1, num_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_edges2, edges2, num_bytes, cudaMemcpyHostToDevice); cudaMemcpy(d_edges3, edges3, num_bytes, cudaMemcpyHostToDevice);
	gb2<<<grid_size,block_size>>>(width, height, c, d_edges_remain0, d_edges_remain1, d_edges_remain2, d_edges_remain3,
                                      d_u0, d_u1, d_u2, d_u3, d_er_num, d_edges0, d_edges1, d_edges2, d_edges3);
// 	check_cuda_errors(__FILE__, __LINE__); 
	err = cudaGetLastError();
	printf("Error: %s\n", cudaGetErrorString(err) );

	err5 = cudaGetErrorString(cudaPeekAtLastError());
	printf("-------------------------------------%s\n", err5);
	err5 = cudaGetErrorString(cudaThreadSynchronize());
	printf("-------------------------------------%s\n", err5);	
	printf("End segmenting graph in GPU.\n");

	err3 = cudaMemcpy(u0, d_u0, num_bytes_n, cudaMemcpyDeviceToHost); cudaMemcpy(u1, d_u1, num_bytes_n, cudaMemcpyDeviceToHost);
	cudaMemcpy(u2, d_u2, num_bytes_n, cudaMemcpyDeviceToHost); cudaMemcpy(u3, d_u3, num_bytes_n, cudaMemcpyDeviceToHost);
	printf("Error3: %s\n", cudaGetErrorString(err3) );
	for (int i = 0; i < num_vertices; ++i) 
          mess->set_in_level(i, level, u0->find(i), u0->rank(i), u0->size(i), u0->mst(i)); 
        for (int i = num_vertices; i < 2*num_vertices; ++i) 
          mess->set_in_level(i, level, u1->find(i-num_vertices), u1->rank(i-num_vertices), u1->size(i-num_vertices), u1->mst(i-num_vertices));
        for (int i = 2*num_vertices; i < 3*num_vertices; ++i) 
          mess->set_in_level(i, level, u2->find(i-2*num_vertices), u2->rank(i-2*num_vertices), u2->size(i-2*num_vertices), u2->mst(i-2*num_vertices));
        for (int i = 3*num_vertices; i < 4*num_vertices; ++i) 
          mess->set_in_level(i, level, u3->find(i-3*num_vertices), u3->rank(i-3*num_vertices), u3->size(i-3*num_vertices), u3->mst(i-3*num_vertices));
       
	/*err5 =*/ cudaMemcpy(er_num, d_er_num, 4*sizeof(int), cudaMemcpyDeviceToHost);
//	printf("Error5: %s\n", cudaGetErrorString(err5) );
	for (int i = 0; i < 4; ++i) {
          printf("edges_remain %d has %d elements.\n", i, er_num[i]);
        }
 
	// output oversegmentation in level 0 of heirarchical system 
        generate_output_s(path, num_frame, width, height, u0, num_vertices, 0); 
        generate_output_s(path, num_frame, width, height, u1, num_vertices, 1); 
        generate_output_s(path, num_frame, width, height, u2, num_vertices, 2); 
        generate_output_s(path, num_frame, width, height, u3, num_vertices, 3); 

	// transfter edges to edges_remian for first level hierarchical segmentation	
        err4 = cudaMemcpy(edges_remain0, d_edges_remain0, num_bytes, cudaMemcpyDeviceToHost);
	printf("Error4: %s\n", cudaGetErrorString(err4) );
	cudaMemcpy(edges_remain1, d_edges_remain1, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(edges_remain2, d_edges_remain2, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(edges_remain3, d_edges_remain3, num_bytes, cudaMemcpyDeviceToHost);

	// collect remained edges which were not merged in first level graph-based segmentation
        for ( int it = 0; it < er_num[0]; it++ )
          edges_remain->push_back(edges_remain0[it]); 
        for ( int it = 0; it < er_num[1]; it++ )
          edges_remain->push_back(edges_remain1[it]); 
        for ( int it = 0; it < er_num[2]; it++ )
          edges_remain->push_back(edges_remain2[it]); 
        for ( int it = 0; it < er_num[3]; it++ )
          edges_remain->push_back(edges_remain3[it]); 
                
	sort(edges_remain->begin(), edges_remain->end());
	printf("Edges region0 number is %d.\n", (int)edges_remain->size());

	// clear temporary variables
        delete edges_remain0; cudaFree(d_edges_remain0); delete edges_remain1; cudaFree(d_edges_remain1);
        delete edges_remain2; cudaFree(d_edges_remain2); delete edges_remain3; cudaFree(d_edges_remain3);
	cudaFree(d_u0); cudaFree(d_u1); cudaFree(d_u2); cudaFree(d_u3);
	cudaFree(d_er_num);  
//        cudaFree(d_smooth_r); cudaFree(d_smooth_g); cudaFree(d_smooth_b);
	cudaFree(d_edges0); cudaFree(d_edges1); cudaFree(d_edges2); cudaFree(d_edges3); 
//	delete t_smooth_r; delete t_smooth_g; delete t_smooth_b; 
	free(t_smooth_r); free(t_smooth_g); free(t_smooth_b); 
}

/* Gaussian Smoothing */
void smooth_images(image<rgb> *im[], int num_frame, image<float> *smooth_r[],
		image<float> *smooth_g[], image<float> *smooth_b[], float sigma) {

	int width = im[0]->width();
	int height = im[0]->height();

	image<float>** r = new image<float>*[num_frame];
	image<float>** g = new image<float>*[num_frame];
	image<float>** b = new image<float>*[num_frame];
	#pragma omp parallel for 
	for (int i = 0; i < num_frame; i++) {
		r[i] = new image<float>(width, height);
		g[i] = new image<float>(width, height);
		b[i] = new image<float>(width, height);
	}
	for (int i = 0; i < num_frame; i++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				imRef(r[i], x, y) = imRef(im[i], x, y).r;
				imRef(g[i], x, y) = imRef(im[i], x, y).g;
				imRef(b[i], x, y) = imRef(im[i], x, y).b;
			}
		}
	}
	// smooth each color channel
//	#pragma omp parallel for 
	for (int i = 0; i < num_frame; i++) {
		smooth_r[i] = smooth(r[i], sigma);
		smooth_g[i] = smooth(g[i], sigma);
		smooth_b[i] = smooth(b[i], sigma);
	}
	#pragma omp parallel for 
	for (int i = 0; i < num_frame; i++) {
		delete r[i];
		delete g[i];
		delete b[i];
	}
	delete[] r;
	delete[] g;
	delete[] b;
}

/* Save Output */
void generate_output(char *path, int num_frame, int width, int height,
		universe *mess, int num_vertices, int level_total) {

	char savepath[1024];
	image<rgb>** output = new image<rgb>*[num_frame];
	rgb* colors = new rgb[num_vertices];
	for (int i = 0; i < num_vertices; i++)
		colors[i] = random_rgb();

	// write out the ppm files.
	for (int k = 0; k <= level_total; k++) {
		for (int i = 0; i < num_frame; i++) {
			// output 1 higher level than them in GBH and replace k with k+1
			snprintf(savepath, 1023, "%s/%02d/%05d.ppm", path, k+1, i + 1);
			output[i] = new image<rgb>(width, height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int comp = mess->find_in_level(
							y * width + x + i * (width * height), k);
					imRef(output[i], x, y) = colors[comp];
				}
			}
			savePPM(output[i], savepath);
		}
		#pragma omp parallel for 
		for (int i = 0; i < num_frame; i++)
			delete output[i];
	}
	delete[] colors;
	delete[] output;

}

/* main operation steps */
void segment_image(char *path, image<rgb> *im[], int num_frame, float c,
		float c_reg, int min_size, float sigma, int hie_num) {

	// step 1 -- Get information
	int width = im[0]->width();
	int height = im[0]->height();

	// ----- node number
	int num_vertices = num_frame * width * height;
	// ----- edge number
	int num_edges_plane = (width - 1) * (height - 1) * 2 + width * (height - 1)
			+ (width - 1) * height;
	int num_edges_layer = (width - 2) * (height - 2) * 9 + (width - 2) * 2 * 6
			+ (height - 2) * 2 * 6 + 4 * 4;
	int num_edges = num_edges_plane * num_frame
			+ num_edges_layer * (num_frame - 1);

	// ----- hierarchy setup
	vector<vector<edge>*> edges_region;
	edges_region.resize(hie_num + 1);

	// ------------------------------------------------------------------

	// step 2 -- smooth images
	image<float>** smooth_r = new image<float>*[num_frame];
	image<float>** smooth_g = new image<float>*[num_frame];
	image<float>** smooth_b = new image<float>*[num_frame];
	smooth_images(im, num_frame, smooth_r, smooth_g, smooth_b, sigma);
	// ------------------------------------------------------------------

	// step 3 -- build edges
	printf("start build edges\n");
	edge* edges = new edge[num_edges];
	initialize_edges(edges, num_frame, width, height, smooth_r, smooth_g,
			smooth_b, 0);
	printf("end build edges\n");
	// ------------------------------------------------------------------
	printf("The edges' number is %d.\n", num_edges);
	// step 4 -- build nodes
	printf("start build nodes\n");
	universe* mess = new universe(num_frame, width, height, smooth_r, smooth_g,
			smooth_b, hie_num);
	printf("end build nodes\n");
	// ------------------------------------------------------------------

	// step 5 -- over-segmentation
	printf("start over-segmentation\n");
	edges_region[0] = new vector<edge>();
	segment_graph(mess, edges_region[0], edges, c, width, height, 0,
                      smooth_r, smooth_g, smooth_b, num_frame/num_cores, path);

	// optional merging small components
/*	for (int i = 0; i < num_edges; i++) {
		int a = mess->find_in_level(edges[i].a, 0);
		int b = mess->find_in_level(edges[i].b, 0);
		if ((a != b)
				&& ((mess->get_size(a) < min_size)
						|| (mess->get_size(b) < min_size)))
			mess->join(a, b, 0, 0);
	}
	printf("end over-segmentation\n");
	// ------------------------------------------------------------------
*/

	// step 6 -- hierarchical segmentation
	for (int i = 0; i < hie_num; i++) {
		printf("level = %d\n", i);
		// incremental in each hierarchy
		min_size = min_size * 1.2;

		printf("start update\n");
		mess->update(i);
		printf("end update\n");

		printf("start fill edge weight\n");
		fill_edge_weight(*edges_region[i], mess, i);
		printf("end fill edge weight\n");

		printf("Edges region%d number is %d.\n", i, (int)edges_region[i]->size());
		printf("start segment graph region\n");
		edges_region[i + 1] = new vector<edge>();
		segment_graph_region(mess, edges_region[i + 1], edges_region[i], c_reg, i + 1);
		printf("end segment graph region\n");

		printf("start merging min_size\n");
		for (int it = 0; it < (int) edges_region[i]->size(); it++) {
			int a = mess->find_in_level((*edges_region[i])[it].a, i + 1);
			int b = mess->find_in_level((*edges_region[i])[it].b, i + 1);
			if ((a != b)
					&& ((mess->get_size(a) < min_size)
							|| (mess->get_size(b) < min_size)))
				mess->join(a, b, 0, i + 1);
		}
		printf("end merging min_size\n");

		c_reg = c_reg * 1.4;
		delete edges_region[i];
	}
	delete edges_region[hie_num];
	// ------------------------------------------------------------------

	// step 8 -- generate output
	printf("start output\n");
	generate_output(path, num_frame, width, height, mess, num_vertices,
			hie_num);
	printf("end output\n");
	// ------------------------------------------------------------------

	// step 9 -- clear everything
	delete mess;
	delete[] edges;
	#pragma omp parallel for 
	for (int i = 0; i < num_frame; i++) {
		delete smooth_r[i];
		delete smooth_g[i];
		delete smooth_b[i];
	}
	delete[] smooth_r;
	delete[] smooth_g;
	delete[] smooth_b;

}

int main(int argc, char **argv) {
	if (argc != 8) {
		printf("%s c c_reg min sigma hie_num input output\n", argv[0]);
		printf("       c --> value for the threshold function in over-segmentation\n");
		printf("   c_reg --> value for the threshold function in hierarchical region segmentation\n");
		printf("     min --> enforced minimum supervoxel size\n");
		printf("   sigma --> variance of the Gaussian smoothing.\n");
		printf(" hie_num --> desired number of hierarchy levels\n");
		printf("   input --> input path of ppm video frames\n");
		printf("  output --> output path of segmentation results\n");
		return 1;
	}

	// Read Parameters
	float c = atof(argv[1]);
	float c_reg = atof(argv[2]);
	int min_size = atoi(argv[3]);
	float sigma = atof(argv[4]);
	int hie_num = atoi(argv[5]);
	char* input_path = argv[6];
	char* output_path = argv[7];
	if (c <= 0 || c_reg < 0 || min_size < 0 || sigma < 0 || hie_num < 0) {
		fprintf(stderr, "Unable to use the input parameters.");
		return 1;
	}

	// count files in the input directory
	int frame_num = 0;
	struct dirent* pDirent;
	DIR* pDir;
	pDir = opendir(input_path);
	if (pDir != NULL) {
		while ((pDirent = readdir(pDir)) != NULL) {
			int len = strlen(pDirent->d_name);
			if (len >= 4) {
				if (strcmp(".ppm", &(pDirent->d_name[len - 4])) == 0)
					frame_num++;
			}
		}
	}
	if (frame_num == 0) {
		fprintf(stderr, "Unable to find video frames at %s", input_path);
		return 1;
	}
	printf("Total number of frames in fold is %d\n", frame_num);


	// make the output directory
	struct stat st;
	int status = 0;
	char savepath[1024];
  	snprintf(savepath,1023,"%s",output_path);
	if (stat(savepath, &st) != 0) {
		/* Directory does not exist */
		if (mkdir(savepath, S_IRWXU) != 0) {
			status = -1;
		}
	}
	for (int i = 0; i <= (hie_num+1); i++) {
  		snprintf(savepath,1023,"%s/%02d",output_path,i);
		if (stat(savepath, &st) != 0) {
			/* Directory does not exist */
			if (mkdir(savepath, S_IRWXU) != 0) {
				status = -1;
			}
		}
	}
	if (status == -1) {
		fprintf(stderr,"Unable to create the output directories at %s",output_path);
		return 1;
	}


	// Initialize Parameters
	image<rgb>** images = new image<rgb>*[frame_num];
	char filepath[1024];

	// Time Recorder
	time_t Start_t, End_t;
	int time_task;
	Start_t = time(NULL);

	// Read Frames
	for (int i = 0; i < frame_num; i++) {
		snprintf(filepath, 1023, "%s/%05d.ppm", input_path, i + 1);
		images[i] = loadPPM(filepath);
		printf("load --> %s\n", filepath);
	}

	// segmentation
	segment_image(output_path, images, frame_num, c, c_reg, min_size, sigma, hie_num);

	// Time Recorder
	End_t = time(NULL);
	time_task = difftime(End_t, Start_t);
	std::ofstream myfile;
	char timefile[1024];
	snprintf(timefile, 1023, "%s/%s", output_path, "time.txt");
	myfile.open(timefile);
	myfile << time_task << endl;
	myfile.close();

	printf("Congratulations! It's done!\n");
	printf("Time_total = %d seconds\n", time_task);
	return 0;
}

