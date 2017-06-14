#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {
  //fixed size identity matrix used in all calculations
  I_ = MatrixXd::Identity(4, 4);
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  //predict the state
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::UpdateCommon(const VectorXd &y) {
  //common update components from Update() and UpdateEKF()
  MatrixXd PHt = P_ * H_.transpose();
  MatrixXd S = H_ * PHt + R_;
  MatrixXd K = PHt * S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  //update the state by using Kalman Filter equations
  VectorXd y = z - H_ * x_;
  UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //update the state by using Extended Kalman Filter equations
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float rho = sqrt(px*px + py*py);

  //avoid potential divide by zero issues
  if (rho < 0.0001) rho = 0.0001;

  VectorXd z_pred(3);
  z_pred << rho, atan2(py, px), (px*vx + py*vy)/ rho;

  VectorXd y = z - z_pred;

  //fix any angles outside the range of -PI to PI
  while(y(1) < -M_PI) y(1) += 2 * M_PI;
  while(y(1) > M_PI)  y(1) -= 2 * M_PI;

  UpdateCommon(y);
}
