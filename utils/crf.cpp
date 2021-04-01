#include <limits>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

const double neginf = -INFINITY;
const int DUMMY_STATE = -1;

template < class T> inline T _min (T x, T y) {return (x < y) ? x :y;}
template < class T> inline T _max (T x, T y) {return (x > y) ? x :y;}

# define MINUS_LOS_EPSION 50

inline double logsumexp(double arr[], int count) {
  if (count > 0) {
    double maxVal = *max_element(arr, arr + count);
    if (maxVal == neginf) {
      return neginf;
    }
    double sum = 0;
    for (int i = 0; i < count; i++) {
      sum += exp(arr[i] - maxVal);
    }
    return log(sum) + maxVal;
  }
  else
  {
    return 0.0;
  }
}

extern "C" double viterbi_all(
    double *observation, double* transition, int xLen, int yLen, int tpl_length,
    long* argmaxPos, long* new_traceback) {
  int numStates = 3;
  double* score = new double[(xLen+1)*(yLen+1)*(numStates+1)];
  memset(score, 0, (xLen+1)*(yLen+1)*(numStates+1)*sizeof(score[0]));
  double* padding = new double[xLen*yLen*numStates];
  memset(padding, 0, xLen*yLen*numStates*sizeof(padding[0]));

  int* traceback = new int[(xLen+1)*(yLen+1)*(numStates)];
  memset(traceback, DUMMY_STATE, (xLen+1)*(yLen+1)*(numStates)*sizeof(traceback[0]));



  // score[:, 0, :3] = neginf
  // score[0, :, :3] = neginf
  // score[tpl_length+1:xLen+1, 1:yLen+1, :3] = neginf
  // padding[tpl_length:xLen, 0:yLen, :3] = neginf
  for (int m = 0; m < numStates; m++) {
    for (int i = 0; i <= xLen; i++)
      score[i*(yLen+1)*(numStates+1)+m] = neginf;
    for (int j = 0; j <= yLen; j++)
      score[j*(numStates+1)+m] = neginf;
    for (int i = tpl_length; i < xLen; i++) {
      for (int j = 0; j < yLen; j++) {
        padding[i*yLen*numStates+j*numStates+m] = neginf;
        score[(i+1)*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)+m] = neginf;
      }
    }
  }

  for (int i = 1; i < xLen + 1; i++) {
    for (int j = 1; j < yLen + 1; j++) {
      double maxv0 = neginf, maxv1 = neginf, maxv2 = neginf;
      int state0 = DUMMY_STATE, state1 = DUMMY_STATE, state2 =DUMMY_STATE;
      double intermediate;
      // Match
      // score[i, j] = max(score[i-1, j-1, :4] + transition[:4, 0]) + obs[i-1, j-1, 0]
      for (int m=0; m < numStates + 1; m++) {
        intermediate = score[(i-1)*(yLen+1)*(numStates+1)+(j-1)*(numStates+1)+m] + transition[m*5];
        if (intermediate > maxv0)
          maxv0 = intermediate, state0 = m;
      }
      traceback[i*(yLen+1)*(numStates)+j*(numStates)] = state0;
      score[i*(yLen+1)*(numStates+1)+ j*(numStates+1)] = observation[(i-1)*yLen*numStates+(j-1)*numStates] + maxv0 + padding[(i-1)*yLen*numStates+(j-1)*numStates];

      // Insert X
      // score[i, j] = max(score[i-1, j, :3] + transition[:3, 1]) + obs[i-1, j-1, 1]
      for (int m=0; m < numStates; m++) {
        intermediate = score[(i-1)*(yLen+1)*(numStates+1)+j*(numStates+1)+m] + transition[m*5+1];
        if (intermediate > maxv1)
          maxv1 = intermediate, state1 = m;
      }
      traceback[i*(yLen+1)*(numStates)+j*numStates+1] = state1;
      score[i*(yLen+1)*(numStates+1)+ j*(numStates+1)+1] = observation[(i-1)*yLen*numStates+(j-1)*numStates+1] + maxv1 + padding[(i-1)*yLen*numStates+(j-1)*numStates+1];

      // Insert Y
      // score[i, j] = max(score[i, j-1, :3] + transition[:3, 2]) + obs[i-1, j-1, 2]
      for (int m=0; m < numStates; m++) {
        intermediate = score[i*(yLen+1)*(numStates+1)+(j-1)*(numStates+1)+m] + transition[m*5+2];
        if (intermediate > maxv2)
          maxv2 = intermediate, state2 = m;
      }
      traceback[i*(yLen+1)*(numStates)+j*numStates+2] = state2;
      score[i*(yLen+1)*(numStates+1)+ j*(numStates+1)+2] = observation[(i-1)*yLen*(numStates)+(j-1)*(numStates)+2] + maxv2 + padding[(i-1)*yLen*numStates+(j-1)*numStates+2];
    }
  }

  // new_score = score[1:, 1:, 0]
  double* new_score = new double[xLen*yLen];
  for (int i = 0; i< xLen; i++) {
    for (int j = 0; j < yLen; j++)
      new_score[i*yLen+j] = score[(i+1)*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)];
  }

  double maxScore = *max_element(new_score, new_score+xLen*yLen);
  int index_max = max_element(new_score, new_score+xLen*yLen) - new_score;

  int x_pos = index_max / yLen;
  int y_pos = index_max % yLen;
  argmaxPos[0] = x_pos;
  argmaxPos[1] = y_pos;

  for (int i = 0; i < xLen; i++) {
    for (int j = 0; j < yLen; j++) {
      for (int m = 0; m < numStates; m++)
        new_traceback[i*yLen*numStates+j*numStates+m] = traceback[(i+1)*(yLen+1)*numStates+(j+1)*numStates+m];
    }
  }
  delete []new_score; delete []score; delete []padding; delete []traceback;
  return maxScore;
}


extern "C" double crf_forward(
    double* observation, double* transition,
    int xLen, int yLen, int tpl_length, int seq_length, double* alpha) {
  int numStates = 3;
  double* score = new double[(xLen+1)*(yLen+1)*(numStates+1)];
  memset(score, 0, (xLen+1)*(yLen+1)*(numStates+1)*sizeof(score[0]));
  double* padding = new double[xLen*yLen*numStates];
  memset(padding, 0, xLen*yLen*numStates*sizeof(padding[0]));

  // score[0, :, :3] = neginf
  // score[:, 0, :3] = neginf
  for (int m = 0; m < numStates; m++) {
    for (int i = 0; i <= xLen; i++)
      score[i*(yLen+1)*(numStates+1)+m] = neginf;
    for (int j = 0; j <= yLen; j++)
      score[j*(numStates+1)+m] = neginf;
  }

  // score[tpl_length+1:xLen+1, :yLen+1, :3] = neginf
  // score[:xLen+1, tgt_length+1:yLen+1, :3] = neginf
  // padding[tpl_length:xLen, :yLen, :3] = neginf
  // padding[:xLen, tgt_length:yLen, :3] = neginf
  for (int m = 0; m < numStates; m++) {
    for (int i = tpl_length; i < xLen; i++) {
      for (int j = 0; j < yLen; j++) {
        score[(i+1)*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)+m] = neginf;
        padding[i*yLen*numStates+j*numStates+m] = neginf;
      }
    }
    for (int j = seq_length; j < yLen; j++) {
      for (int i = 0; i < xLen; i++) {
        score[(i+1)*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)+m] = neginf;
        padding[i*yLen*numStates+j*numStates+m] = neginf;
      }
    }
  }

  for (int j = 1; j < yLen + 1; j++) {
    for (int i = 1; i < xLen + 1; i++) {
      double intermediate;
      double* state_score = new double[numStates+1];

      // Match
      // score = log_sum_exp(alpha[i-1, j-1, :4] + transition[:4, 0]) + obs
      memset(state_score, 0, (numStates+1)*sizeof(state_score[0]));
      for (int m = 0; m < numStates + 1; m++)
        state_score[m] = score[(i-1)*(yLen+1)*(numStates+1)+(j-1)*(numStates+1)+m] + transition[m*5];
      intermediate = logsumexp(state_score, numStates+1);
      score[i*(yLen+1)*(numStates+1)+ j*(numStates+1)] = observation[(i-1)*yLen*numStates+(j-1)*numStates] + intermediate + padding[(i-1)*yLen*numStates+(j-1)*numStates];

      // Insert Y
      // score = logsumexp(alpha[i, j-1, :3] + transition[:3, 2]) + obs
      memset(state_score, 0, (numStates+1)*sizeof(state_score[0]));
      for (int m = 0; m < numStates; m++)
        state_score[m] = score[i*(yLen+1)*(numStates+1)+(j-1)*(numStates+1)+m] + transition[m*5+2];
      intermediate = logsumexp(state_score, numStates);
      score[i*(yLen+1)*(numStates+1)+ j*(numStates+1)+2] = observation[(i-1)*yLen*numStates+(j-1)*numStates+2] + intermediate + padding[(i-1)*yLen*numStates+(j-1)*numStates+2];

      // Insert X
      // score = logsumexp(alpha[i-1, j, :3] + transition[:3, 1]) + obs
      memset(state_score, 0, (numStates+1)*sizeof(state_score[0]));
      for (int m = 0; m < numStates; m++)
        state_score[m] = score[(i-1)*(yLen+1)*(numStates+1)+j*(numStates+1)+m] + transition[m*5+1];
      intermediate = logsumexp(state_score, numStates);
      score[i*(yLen+1)*(numStates+1)+ j*(numStates+1)+1] = observation[(i-1)*yLen*numStates+(j-1)*numStates+1] + intermediate + padding[(i-1)*yLen*numStates+(j-1)*numStates+1];

      delete []state_score;
    }
  }

  for (int i = 0; i < xLen; i++) {
    for (int j = 0; j < yLen; j++) {
      for (int m = 0; m < numStates; m++)
        alpha[i*yLen*numStates+j*numStates+m] = score[(i+1)*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)+m];
    }
  }

  double* alpha_score = new double[xLen*yLen];
  for (int i = 0; i < xLen; i++) {
    for (int j = 0; j < yLen; j++) {
      alpha_score[i*yLen+j] = alpha[i*yLen*numStates+j*numStates];
    }
  }

  delete []score; delete []padding;
  double partition = logsumexp(alpha_score, xLen*yLen);
  delete []alpha_score;

  return partition;
}



extern "C" double crf_backward(
    double* observation, double* transition,
    int xLen, int yLen, int tpl_length, int seq_length, double* beta) {
  int numStates = 3;
  double* score = new double[(xLen+1)*(yLen+1)*(numStates+1)];
  memset(score, 0, (xLen+1)*(yLen+1)*(numStates+1)*sizeof(score[0]));
  double* padding = new double[xLen*yLen*numStates];
  memset(padding, 0, xLen*yLen*numStates*sizeof(padding[0]));

  // score[xLen, :, :3] = neginf
  // score[:, yLen, :3] = neginf
  for (int m = 0; m < numStates; m++) {
    for (int i = 0; i <= xLen; i++)
      score[i*(yLen+1)*(numStates+1)+yLen*(numStates+1)+m] = neginf;
    for (int j = 0; j <= yLen; j++)
      score[xLen*(yLen+1)*(numStates+1)+j*(numStates+1)+m] = neginf;
  }

  // score[tpl_length:xLen, :yLen, :3] = neginf
  // score[:xLen, tgt_length:yLen, :3] = neginf
  // padding[tpl_length:xLen, :yLen, :3] = neginf
  // padding[:xLen, tgt_length:yLen, :3] = neginf
  for (int m = 0; m < numStates; m++) {
    for (int i = tpl_length; i < xLen; i++) {
      for (int j = 0; j < yLen; j ++) {
        score[i*(yLen+1)*(numStates+1)+j*(numStates+1)+m] = neginf;
        padding[i*yLen*numStates+j*numStates+m] = neginf;
      }
    }
    for (int j = seq_length; j < yLen; j ++) {
      for (int i = 0; i < xLen; i++) {
        score[i*(yLen+1)*(numStates+1)+j*(numStates+1)+m] = neginf;
        padding[i*yLen*numStates+j*numStates+m] = neginf;
      }
    }
  }

  for (int j = yLen - 1; j >= 0; j--) {
    for (int i = xLen - 1; i >= 0; i--) {
      double intermediate;
      double* state_score = new double[numStates+1];

      // Insert X
      // score = log_sum_exp(beta[i+1, j+1, 0] + transition[1, 0],
      //                     beta[i, j+1, 2] + transition[1, 2],
      //                     beta[i+1, j, 1] + transition[1, 1]) + obs
      memset(state_score, 0, (numStates+1)*sizeof(state_score[0]));
      state_score[0] = score[(i+1)*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)] + transition[1*5];
      state_score[2] = score[i*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)+2] + transition[1*5+2];
      state_score[1] = score[(i+1)*(yLen+1)*(numStates+1)+j*(numStates+1)+1] + transition[1*5+1];
      intermediate = logsumexp(state_score, numStates);

      score[i*(yLen+1)*(numStates+1)+j*(numStates+1)+1] = observation[i*yLen*numStates+j*numStates+1] + intermediate + padding[i*yLen*numStates+j*numStates+1];

      // Match
      // score = logsumexp(beta[i+1, j+1, 0] + transition[0, 0],
      //                   beta[i, j+1, 2] + transition[0, 2],
      //                   beta[i+1, j, 1] + transition[0, 1],
      //                   beta[i+1, j+1, 3] + transition[0, 4]) + obs
      memset(state_score, 0, (numStates+1)*sizeof(state_score[0]));
      state_score[0] = score[(i+1)*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)] + transition[0];
      state_score[2] = score[i*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)+2] + transition[2];
      state_score[1] = score[(i+1)*(yLen+1)*(numStates+1)+j*(numStates+1)+1] + transition[1];
      state_score[3] = score[(i+1)*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)+3] + transition[4];
      intermediate = logsumexp(state_score, numStates+1);
      score[i*(yLen+1)*(numStates+1)+j*(numStates+1)] = observation[i*yLen*numStates+j*numStates] + intermediate + padding[i*yLen*numStates+j*numStates];

      // Insert Y
      // score = logsumexp(beta[i+1, j+1, 0] + transition[2, 0],
      //                   beta[i, j+1, 2] + transition[2, 2],
      //                   beta[i+1, j, 1] + transition[2, 1]) + obs
      memset(state_score, 0, (numStates+1)*sizeof(state_score[0]));
      state_score[0] = score[(i+1)*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)] + transition[2*5];
      state_score[2] = score[i*(yLen+1)*(numStates+1)+(j+1)*(numStates+1)+2] + transition[2*5+2];
      state_score[1] = score[(i+1)*(yLen+1)*(numStates+1)+j*(numStates+1)+1] + transition[2*5+1];
      intermediate = logsumexp(state_score, numStates);
      score[i*(yLen+1)*(numStates+1)+j*(numStates+1)+2] = observation[i*yLen*numStates+j*numStates+2] + intermediate + padding[i*yLen*numStates+j*numStates+2];

      delete []state_score;
    }
  }

  for (int i = 0; i < xLen; i++) {
    for (int j = 0; j < yLen; j++) {
      for (int m = 0; m < numStates; m++)
        beta[i*yLen*numStates+j*numStates+m] = score[i*(yLen+1)*(numStates+1)+j*(numStates+1)+m];
    }
  }

  double* beta_score = new double[xLen*yLen];
  for (int i = 0; i < xLen; i++) {
    for (int j = 0; j < yLen; j++) {
      beta_score[i*yLen+j] = beta[i*yLen*numStates+j*numStates];
    }
  }

  delete []score; delete []padding;
  double partition = logsumexp(beta_score, xLen*yLen);
  delete []beta_score;

  return partition;
}
