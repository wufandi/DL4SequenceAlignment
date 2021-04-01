# include <stdio.h>
# include <string.h>
# include <string>
# include <stdint.h>
# include <cmath>

using namespace std;

const double edge_potent_d_ref[12] = {0.001401900,0.003644220,0.004355920,0.004943320,0.008613160,0.015290900,0.021347600,0.019730100,0.022954800,0.024599900,0.028235800,00.844883000};
const double MATCH_BOUND_UPPER = 15;
const double MATCH_BOUND_LOWER = -5;

double DistOf2AAs(double *atom_i, double *atom_j) {
  return sqrt((atom_i[0]-atom_j[0])*(atom_i[0]-atom_j[0]) + \
    (atom_i[1]-atom_j[1])*(atom_i[1]-atom_j[1]) + \
    (atom_i[2]-atom_j[2])*(atom_i[2]-atom_j[2]));
}


double compute_edge_potential(
    int xLen, int yLen, int method_length,
    int ti, int tj, int qi, int qj, double* pair_dis, double* dis_matrix,
    double* disc_method, bool use_dis_pot) {
  double pot = 0;
  int t_index = 0;
  double dis = dis_matrix[ti * xLen + tj];
  t_index = method_length-1;
  for (int i = 0; i < method_length - 1; i++) {
    if (disc_method[i] < dis && dis <= disc_method[i+1])
      t_index = i;
  }
  if (use_dis_pot == 1)
    pot = -pair_dis[qi * yLen * method_length + qj * method_length + t_index];
  else {
    pot = log(pair_dis[qi * yLen * method_length + \
        qj * method_length + t_index]) / edge_potent_d_ref[t_index];
    if (pot > MATCH_BOUND_UPPER)
      pot = MATCH_BOUND_UPPER;
    if (pot < MATCH_BOUND_UPPER)
      pot = MATCH_BOUND_LOWER;
  }
  return pot;
}


extern "C" void compute_distance(
    char sequence[], int tpl_length, long* missing, double* CA, double* CB,
    double* dis_matrix) {
  memset(dis_matrix, 100, tpl_length*tpl_length*sizeof(dis_matrix[0]));
  for (int i = 0; i < tpl_length; i++) {
    for (int j = i+1; j < tpl_length; i++) {
      if (missing[i] == 0 && missing[j] == 0) {
        double* atom_i = new double[3];
        double* atom_j = new double[3];
        if (sequence[i] == 'G') {
          atom_i[0] = CA[(i-1)*3];
          atom_i[1] = CA[(i-1)*3+1];
          atom_i[2] = CA[(i-1)*3+2];
        }
        else {
          atom_i[0] = CB[(i-1)*3];
          atom_i[1] = CB[(i-1)*3+1];
          atom_i[2] = CB[(i-1)*3+2];
        }
        if (sequence[j] == 'G') {
          atom_j[0] = CA[(j-1)*3];
          atom_j[1] = CA[(j-1)*3+1];
          atom_j[2] = CA[(j-1)*3+2];
        }
        else {
          atom_j[0] = CB[(j-1)*3];
          atom_j[1] = CB[(j-1)*3+1];
          atom_j[2] = CB[(j-1)*3+2];
        }
      dis_matrix[i*tpl_length+j] = DistOf2AAs(atom_i, atom_j);
      dis_matrix[j*tpl_length+i] = dis_matrix[i*tpl_length+j];
      delete []atom_i; delete []atom_j;
      }
    }
  }
}


extern "C" void compute_distance_score(
    long* alignment, double* pair_dis, double* dis_matrix, double* disc_method,
    double Norm_Weight, int use_dis_pot,
    int xLen, int yLen, int alignLen, int method_length, double* score) {
  long x1_pos, y1_pos, state1;
  long x2_pos, y2_pos, state2;
  double sum_pair_pot = 0;
  double sum_pair_num = 0;
  for (int i = 0; i < alignLen; i++) {
    x1_pos = alignment[3*i];
    y1_pos = alignment[3*i+1];
    state1 = alignment[3*i+2];
    sum_pair_pot = 0;
    sum_pair_num = 0;
    if (state1 == 0) {
      for (int j = 0; j < alignLen; j++) {
        x2_pos = alignment[3*j];
        y2_pos = alignment[3*j+1];
        state2 = alignment[3*j+2];
        if (state2 == 0 && abs(x2_pos-x1_pos) >= 6 && abs(y2_pos-y1_pos) >= 6 && (x1_pos-x2_pos) * (y1_pos-y2_pos) > 0) {
          double score = compute_edge_potential(
              xLen, yLen, method_length,
              x1_pos, x2_pos, y1_pos, y2_pos, pair_dis,
              dis_matrix, disc_method, use_dis_pot);
          sum_pair_pot += score;
          sum_pair_num += 1;
        }
      }
      if (use_dis_pot == 1) {
        score[0] += sum_pair_pot;
        if (sum_pair_pot != 0)
          score[1] += sum_pair_pot / sum_pair_num * Norm_Weight;
      } else {
        if (sum_pair_pot != 0) {
          score[0] += sum_pair_pot / sum_pair_num;
          score[1] += sum_pair_pot / sum_pair_num * Norm_Weight;
        }
      }
    }
  }
}
