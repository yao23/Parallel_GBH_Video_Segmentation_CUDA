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

#define num_cores 4 
#define num_edges_s 3088836 

using namespace std;

/* Save Output for oversegmentation*/
/*void generate_output_s(char *path, int num_frame, int width, int height,
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
*/
// process every image with graph-based segmentation
__global__ void gb(image<float> *smooth_r[], image<float> *smooth_g[], image<float> *smooth_b[],
        int width, int height, float c, edge *edges_remain0[], edge *edges_remain1[], edge *edges_remain2[], edge *edges_remain3[],
        Edge edges0, Edge edges1, Edge edges2, Edge edges3, float *threshold0, float *threshold1,
        float *threshold2, float *threshold3, universe_s *u0, universe_s *u1, universe_s *u2, universe_s *u3) {
  int case_num = blockIdx.x;
  int num_frame = blockDim.x;
  // ----- node number
  int num_vertices = num_frame * width * height;
  switch(case_num) {
    case 0: 
    {
      initialize_edges(edges0, num_frame, width, height, smooth_r, smooth_g, smooth_b, 0);
      //  printf("Finished edge initialization.\n");
      segment_graph_s(num_vertices, num_edges_s, edges0, c, edges_remain0, u0, threshold0);
      //  printf("Finished unit graph segmentation.\n"); 
    }
    break;
    case 1: 
    {
      initialize_edges(edges1, num_frame, width, height, smooth_r, smooth_g, smooth_b, 1);
      //  printf("Finished edge initialization.\n");
      segment_graph_s(num_vertices, num_edges_s, edges1, c, edges_remain1, u1, threshold1);
      //  printf("Finished unit graph segmentation.\n"); 
    }
    break;
    case 2: 
    {
      initialize_edges(edges2, num_frame, width, height, smooth_r, smooth_g, smooth_b, 2);
      //  printf("Finished edge initialization.\n");
      segment_graph_s(num_vertices, num_edges_s, edges2, c, edges_remain2, u2, threshold2);
      //  printf("Finished unit graph segmentation.\n"); 
    }
    break;
    case 3: 
    {
      initialize_edges(edges3, num_frame, width, height, smooth_r, smooth_g, smooth_b, 3);
      //  printf("Finished edge initialization.\n");
      segment_graph_s(num_vertices, num_edges_s, edges3, c, edges_remain3, u3, threshold3);
      //  printf("Finished unit graph segmentation.\n"); 
    }
    break;
    default: break;
  }
//  printf("Finished mess assignment.\n");
}

/* pixel level minimum spanning tree merge */
void segment_graph(universe *mess, vector<edge>* edges_remain, edge *edges, float c, int width, int height, int level,
                image<float> *smooth_r[], image<float> *smooth_g[], image<float> *smooth_b[], int num_frame, char *path) {
	// new vector containing remain edges
	edges_remain->clear();
	printf("Start segmenting graph in parallel.\n");

        int num_vertices = num_frame * width * height;
        int num_bytes = num_edges_s * sizeof(edge); // edge array size
  	int num_bytes_th = num_vertices * sizeof(float); // threshold array size
        int num_bytes_n = num_vertices * sizeof(uni_elt);
	
	int block_size = num_frame;
	int grid_size = num_cores;
	// initialize edges and remained edges array	
	edge **d_edges_remain0 = NULL;  /*edge *d_edges0 = NULL;*/ Edge d_edges0;
	edge **d_edges_remain1 = NULL;  /*edge *d_edges1 = NULL;*/ Edge d_edges1;
	edge **d_edges_remain2 = NULL;  /*edge *d_edges2 = NULL;*/ Edge d_edges2;
	edge **d_edges_remain3 = NULL;  /*edge *d_edges3 = NULL;*/ Edge d_edges3;
	// cudaMalloc memory space for edge vectors 
        cudaMalloc((void**)&d_edges_remain0, num_bytes);  cudaMalloc((void**)&d_edges0, num_bytes);
        cudaMalloc((void**)&d_edges_remain1, num_bytes);  cudaMalloc((void**)&d_edges1, num_bytes);
        cudaMalloc((void**)&d_edges_remain2, num_bytes);  cudaMalloc((void**)&d_edges2, num_bytes);
        cudaMalloc((void**)&d_edges_remain3, num_bytes);  cudaMalloc((void**)&d_edges3, num_bytes);
        // initialize threshold and node array 
   	float *d_th0 = NULL;   	float *d_th1 = NULL;   	float *d_th2 = NULL;   	float *d_th3 = NULL;
        universe_s *d_u0 = new universe_s(num_vertices); universe_s *d_u1 = new universe_s(num_vertices); 
	universe_s *d_u2 = new universe_s(num_vertices); universe_s *d_u3 = new universe_s(num_vertices);
        // allocate memory space for threshold and node array 
        cudaMalloc((void**)&d_th0, num_bytes_th); cudaMalloc((void**)&d_th1, num_bytes_th);
        cudaMalloc((void**)&d_th2, num_bytes_th); cudaMalloc((void**)&d_th3, num_bytes_th);
        cudaMalloc((void**)&d_u0, num_bytes_n); cudaMalloc((void**)&d_u1, num_bytes_n);
        cudaMalloc((void**)&d_u2, num_bytes_n); cudaMalloc((void**)&d_u3, num_bytes_n);
        
	gb<<<grid_size,block_size>>>(smooth_r, smooth_g, smooth_b, width, height, c, 
             d_edges_remain0, d_edges_remain1, d_edges_remain2, d_edges_remain3,
             d_edges0, d_edges1, d_edges2, d_edges3, d_th0, d_th1, d_th2, d_th3, d_u0, d_u1, d_u2, 
             d_u3);
  	
        universe_s *u0 = new universe_s(num_vertices); universe_s *u1 = new universe_s(num_vertices); 
	universe_s *u2 = new universe_s(num_vertices); universe_s *u3 = new universe_s(num_vertices);
       	cudaMemcpy(u0, d_u0, num_bytes_n, cudaMemcpyDeviceToHost); cudaMemcpy(u1, d_u1, num_bytes_n, cudaMemcpyDeviceToHost);
	cudaMemcpy(u2, d_u2, num_bytes_n, cudaMemcpyDeviceToHost); cudaMemcpy(u3, d_u3, num_bytes_n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < num_vertices; ++i) 
          mess->set_in_level(i, level, u0->find(i), u0->rank(i), u0->size(i), u0->mst(i)); 
        for (int i = num_vertices; i < 2*num_vertices; ++i) 
          mess->set_in_level(i, level, u1->find(i-num_vertices), u1->rank(i-num_vertices), u1->size(i-num_vertices), u1->mst(i-num_vertices));
        for (int i = 2*num_vertices; i < 3*num_vertices; ++i) 
          mess->set_in_level(i, level, u2->find(i-2*num_vertices), u2->rank(i-2*num_vertices), u2->size(i-2*num_vertices), u2->mst(i-2*num_vertices));
        for (int i = 3*num_vertices; i < 4*num_vertices; ++i) 
          mess->set_in_level(i, level, u3->find(i-3*num_vertices), u3->rank(i-3*num_vertices), u3->size(i-3*num_vertices), u3->mst(i-3*num_vertices));
        
	// output oversegmentation in level 0 of heirarchical system 
/*        generate_output_s(path, num_frame, width, height, u0, num_vertices, 0); 
        generate_output_s(path, num_frame, width, height, u1, num_vertices, 1); 
        generate_output_s(path, num_frame, width, height, u2, num_vertices, 2); 
        generate_output_s(path, num_frame, width, height, u3, num_vertices, 3); 
        generate_output_s(path, num_frame, width, height, u4, num_vertices, 4); 
        generate_output_s(path, num_frame, width, height, u5, num_vertices, 5); 
        generate_output_s(path, num_frame, width, height, u6, num_vertices, 6); 
        generate_output_s(path, num_frame, width, height, u7, num_vertices, 7); 
*/	// transfter edges to edges_remian for first level hierarchical segmentation	
	edge *edges_remain0 = new edge[num_edges_s];  cudaMemcpy(edges_remain0, d_edges_remain0, num_bytes, cudaMemcpyDeviceToHost);
	edge *edges_remain1 = new edge[num_edges_s];  cudaMemcpy(edges_remain1, d_edges_remain1, num_bytes, cudaMemcpyDeviceToHost);
	edge *edges_remain2 = new edge[num_edges_s];  cudaMemcpy(edges_remain2, d_edges_remain2, num_bytes, cudaMemcpyDeviceToHost);
	edge *edges_remain3 = new edge[num_edges_s];  cudaMemcpy(edges_remain3, d_edges_remain3, num_bytes, cudaMemcpyDeviceToHost);
	// collect remained edges which were not merged in first level graph-based segmentation
        for ( int it = 0; it < num_edges_s; it++ )
          edges_remain->push_back(edges_remain0[it]); 
        for ( int it = 0; it < num_edges_s; it++ )
          edges_remain->push_back(edges_remain1[it]); 
        for ( int it = 0; it < num_edges_s; it++ )
          edges_remain->push_back(edges_remain2[it]); 
        for ( int it = 0; it < num_edges_s; it++ )
          edges_remain->push_back(edges_remain3[it]); 
                
	sort(edges_remain->begin(), edges_remain->end());
	// clear temporary variables
        delete edges_remain0; cudaFree(d_edges_remain0); delete edges_remain1; cudaFree(d_edges_remain1);
        delete edges_remain2; cudaFree(d_edges_remain2); delete edges_remain3; cudaFree(d_edges_remain3);
       	cudaFree(d_th0); cudaFree(d_th1); cudaFree(d_th2); cudaFree(d_th3);
	cudaFree(d_u0); cudaFree(d_u1); cudaFree(d_u2); cudaFree(d_u3);
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
			snprintf(savepath, 1023, "%s/%02d/%05d.ppm", path, k, i + 1);
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
//		initialize_edges(edges, num_frame, width, height, smooth_r, smooth_g, smooth_b, 0);
		printf("end fill edge weight\n");

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
	for (int i = 0; i <= hie_num; i++) {
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

