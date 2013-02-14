/*
Original Code From:
Copyright (C) 2006 Pedro Felzenszwalb
Modifications (may have been made) Copyright (C) 2011,2012 
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

/* Implements node merging criteria. */

#ifndef SEGMENT_GRAPH_MULTI
#define SEGMENT_GRAPH_MULTI

#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>
#include "pnmfile.h"
#include "disjoint-set.h"
#include "disjoint-set-s.h"
#include "segment-graph-s.h"
#include "edges.h"

using namespace std;

// random color
rgb random_rgb() {
	rgb c;
 	c.r = (uchar) random();
        c.g = (uchar) random();
        c.b = (uchar) random();
        return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *r, image<float> *g, image<float> *b,
                         int x1, int y1, int x2, int y2) {
  return sqrt(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
              square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
              square(imRef(b, x1, y1)-imRef(b, x2, y2)));
}

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

// process every image with graph-based segmentation
void gb(universe *mess, image<float> *smooth_r[], image<float> *smooth_g[], image<float> *smooth_b[],
        int width, int height, edge *edges, universe_s *u, float c, int case_num, int level, 
        vector<edge>* edges_remain, int num_edges, int num_frame) {
  
//   printf("The frame number is %d and case number is %d.\n", num_frame, case_num);	
  // ----- node number
  int num_vertices = num_frame * width * height;
  int s_index = case_num * num_vertices;
  int e_index = (case_num + 1) * num_vertices;
//  printf("start and end index are %d and %d.\n", s_index, e_index);
  initialize_edges(edges, num_frame, width, height, smooth_r, smooth_g, smooth_b, case_num);
//  printf("Finished edge initialization.\n");

  u = segment_graph_s(num_vertices, num_edges, edges, c, edges_remain);
//  printf("Finished unit graph segmentation.\n"); 

  for (int i = s_index; i < e_index; ++i) 
    mess->set_in_level(i, level, u->find(i-s_index), u->rank(i-s_index), u->size(i-s_index), u->mst(i-s_index)); 
//  printf("Finished mess assignment.\n");
}

*/
/* pixel level minimum spanning tree merge */
/*void segment_graph(universe *mess, vector<edge>* edges_remain, edge *edges, float c, int width, int height, int level,
                image<float> *smooth_r[], image<float> *smooth_g[], image<float> *smooth_b[], int num_frame, char *path) {
	// new vector containing remain edges
	edges_remain->clear();

	printf("Start segmenting graph in parallel.\n");
	int th_id;
	vector<edge>* edges_remain0 = new vector<edge>();  universe_s* u0 = new universe_s(num_frame * width * height);
	vector<edge>* edges_remain1 = new vector<edge>();  universe_s* u1 = new universe_s(num_frame * width * height);
	vector<edge>* edges_remain2 = new vector<edge>();  universe_s* u2 = new universe_s(num_frame * width * height);
	vector<edge>* edges_remain3 = new vector<edge>();  universe_s* u3 = new universe_s(num_frame * width * height);
	vector<edge>* edges_remain4 = new vector<edge>();  universe_s* u4 = new universe_s(num_frame * width * height);
	vector<edge>* edges_remain5 = new vector<edge>();  universe_s* u5 = new universe_s(num_frame * width * height);
	vector<edge>* edges_remain6 = new vector<edge>();  universe_s* u6 = new universe_s(num_frame * width * height);
	vector<edge>* edges_remain7 = new vector<edge>();  universe_s* u7 = new universe_s(num_frame * width * height);
//	int upxl_num = num_frame * width * height; // unit (10 frames) pixel number
  	// ----- node number
  	int num_vertices = num_frame * width * height;

	// ----- edge number for 1 unit which has 10 video clips
	int num_edges_plane = (width - 1) * (height - 1) * 2 + width * (height - 1) + (width - 1) * height;
        int num_edges_layer = (width - 2) * (height - 2) * 9 + (width - 2) * 2 * 6 + (height - 2) * 2 * 6 + 4 * 4;
        int num_edges = num_edges_plane * num_frame + num_edges_layer * (num_frame - 1);
//        printf("The unit edge number is %d.\n", num_edges);
//        printf("The frame number is %d.\n", num_frame);	
        #pragma omp parallel private(th_id)  
	{
  	  th_id = omp_get_thread_num();
          switch(th_id) {
            case 0: 
            {
//	      int case_num0 = 0;
              edge *edges0 = new edge[num_edges];
              gb(mess, smooth_r, smooth_g, smooth_b, width, height, edges0, u0, c, 0, level, edges_remain0, num_edges, num_frame);
	      delete[] edges0;
            }
	    break;
            case 1: 
            {
//	      int case_num1 = 1;
              edge *edges1 = new edge[num_edges];
              gb(mess, smooth_r, smooth_g, smooth_b, width, height, edges1, u1, c, 1, level, edges_remain1, num_edges, num_frame);            
	      delete[] edges1;
            }
            break;
            case 2: 
       	    {
//	      int case_num2 = 2;
 	      edge *edges2 = new edge[num_edges];
	      gb(mess, smooth_r, smooth_g, smooth_b, width, height, edges2, u2, c, 2, level, edges_remain2, num_edges, num_frame);            
	      delete[] edges2;
            }
            break;
            case 3: 
            {
//	      int case_num3 = 3;
	      edge *edges3 = new edge[num_edges];
	      gb(mess, smooth_r, smooth_g, smooth_b, width, height, edges3, u3, c, 3, level, edges_remain3, num_edges, num_frame);            
	      delete[] edges3;
            }
            break;
            case 4: 
            {
//	      int case_num4 = 4;
	      edge *edges4 = new edge[num_edges];
	      gb(mess, smooth_r, smooth_g, smooth_b, width, height, edges4, u4, c, 4, level, edges_remain4, num_edges, num_frame);            
	      delete[] edges4;
            }
      	    break;
            case 5: 
            {
//	      int case_num5 = 5;
	      edge *edges5 = new edge[num_edges];
              gb(mess, smooth_r, smooth_g, smooth_b, width, height, edges5, u5, c, 5, level, edges_remain5, num_edges, num_frame);            
	      delete[] edges5;
	    }
            break;
            case 6: 
            {
//	      int case_num6 = 6;
	      edge *edges6 = new edge[num_edges];
	      gb(mess, smooth_r, smooth_g, smooth_b, width, height, edges6, u6, c, 6, level, edges_remain6, num_edges, num_frame);            
	      delete[] edges6;
       	    }
       	    break;
      	    case 7: 
       	    {
//	      int case_num7 = 7;
	      edge *edges7 = new edge[num_edges];
	      gb(mess, smooth_r, smooth_g, smooth_b, width, height, edges7, u7, c, 7, level, edges_remain7, num_edges, num_frame);            
	      delete[] edges7;
       	    }
       	    break;
     
       	    default: break;
	  }
    	}
	// output oversegmentation in level 0 of heirarchical system 
        generate_output_s(path, num_frame, width, height, u0, num_vertices, 0); 
        generate_output_s(path, num_frame, width, height, u1, num_vertices, 1); 
        generate_output_s(path, num_frame, width, height, u2, num_vertices, 2); 
        generate_output_s(path, num_frame, width, height, u3, num_vertices, 3); 
        generate_output_s(path, num_frame, width, height, u4, num_vertices, 4); 
        generate_output_s(path, num_frame, width, height, u5, num_vertices, 5); 
        generate_output_s(path, num_frame, width, height, u6, num_vertices, 6); 
        generate_output_s(path, num_frame, width, height, u7, num_vertices, 7); 

	// transfter edges to edges_remian for first level hierarchical segmentation	
	vector<edge>::iterator it;
        for ( it = edges_remain0->begin() ; it < edges_remain0->end(); it++ )
          edges_remain->push_back(*it); 
        for ( it = edges_remain1->begin() ; it < edges_remain1->end(); it++ )
          edges_remain->push_back(*it); 
        for ( it = edges_remain2->begin() ; it < edges_remain2->end(); it++ )
          edges_remain->push_back(*it); 
        for ( it = edges_remain3->begin() ; it < edges_remain3->end(); it++ )
          edges_remain->push_back(*it); 
        for ( it = edges_remain4->begin() ; it < edges_remain4->end(); it++ )
          edges_remain->push_back(*it); 
        for ( it = edges_remain5->begin() ; it < edges_remain5->end(); it++ )
          edges_remain->push_back(*it); 
        for ( it = edges_remain6->begin() ; it < edges_remain6->end(); it++ )
          edges_remain->push_back(*it); 
        for ( it = edges_remain7->begin() ; it < edges_remain7->end(); it++ )
          edges_remain->push_back(*it); 
	sort(edges_remain->begin(), edges_remain->end());

	// clear temporary variables
        delete edges_remain0; delete u0; delete edges_remain1; delete u1;
        delete edges_remain2; delete u2; delete edges_remain3; delete u3;
        delete edges_remain4; delete u4; delete edges_remain5; delete u5;
        delete edges_remain6; delete u6; delete edges_remain7; delete u7;
}
*/	
/* region graph level minimum spanning tree merge */
void segment_graph_region(universe *mess, vector<edge>* edges_remain, 
		vector<edge>* edges_region, float c_reg, int level) {  
	edges_remain->clear();
	sort(edges_region->begin(), edges_region->end());
//	printf("Finished edge region sorting.\n");
//	printf("Edge region size is %d.\n", (int)edges_region->size());
	for (int i=0; i < (int) edges_region->size(); i++) {
//		printf("Operation+1\n");
//		float max_w = 0;
//	        if (max_w < edges_region->at(i).w)
//	          max_w = edges_region->at(i).w;
//	        printf("Maximum region edge weight is %f.\n", max_w); 
		int a = mess->find_in_level(edges_region->at(i).a, level);
		int b = mess->find_in_level(edges_region->at(i).b, level);
//                float a_mst = mess->get_mst(a);
//                float b_mst = mess->get_mst(b);
//                printf("Maximum mst is %f.\n",(a_mst>b_mst?a_mst:b_mst));
		if (a != b) {
			if ((edges_region->at(i).w
					<= mess->get_mst(a) + (c_reg / mess->get_size(a)))
					&& (edges_region->at(i).w
							<= mess->get_mst(b) + (c_reg / mess->get_size(b)))) {
				if (mess->join(a, b, edges_region->at(i).w, level) == 1)
					edges_remain->push_back(edges_region->at(i));
			} else {
				edges_remain->push_back(edges_region->at(i));
			}
//		printf("Operation+1\n");
		}

	}
}
#endif /* SEGMENT_GRAPH_H */
