/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

inline double normalize_angle(double theta) {
	// while (theta < -M_PI) theta += 2 * M_PI;
	// while (theta >  M_PI) theta -= 2 * M_PI;
	return theta;
}

struct _Transform {
	double cos_theta;
	double sin_theta;
	double trans_x;
	double trans_y;
};

inline void map_coord_to_vehicle_coord(const _Transform& trans, const Map::single_landmark_s& landmark, LandmarkObs& out) {
	// c: cos(theta)
	// s: sin(theta)
	// 2d rotation matrix R: 
	// | c -s 0 |
	// | s  c 0 |
	// | 0  0 1 |
	// 2d translation matrix T:
	// | 1 0 tx |
	// | 0 1 ty |
	// | 0 0  1 |
	// R * T:
	// | c, -s, c * tx - s * ty |
	// | s,  c, s * tx + c * ty |
	// | 0,  0,               1 |
	//
	// applying transform:
	// | c, -s, c * tx - s * ty |   | x |   | x * c - y * s + c * tx - s * ty |
	// | s,  c, s * tx + c * ty | * | y | = | x * s + y * c + s * tx + c * ty |
	// | 0,  0,               1 |   | 1 |   |                               1 |
	//
	// above is the math behind this function.
	out.id = landmark.id_i;
	out.x = landmark.x_f * trans.cos_theta - landmark.y_f * trans.sin_theta + trans.cos_theta * trans.trans_x - trans.sin_theta * trans.trans_y;
	out.y = landmark.x_f * trans.sin_theta + landmark.y_f * trans.cos_theta + trans.sin_theta * trans.trans_x + trans.cos_theta * trans.trans_y;
}

inline void vehicle_coord_to_map_coord(const _Transform& trans, const LandmarkObs& local, LandmarkObs& map) {
	// c: cos(theta)
	// s: sin(theta)
	// 2d rotation matrix: 
	// | c -s 0 |
	// | s  c 0 |
	// | 0  0 1 |
	// 2d translation matrix:
	// | 1 0 tx |
	// | 0 1 ty |
	// | 0 0  1 |
	// T * R
	// | c -s tx |
	// | s  c ty |
	// | 0  0  1 |
	//
	// applying transform;
	// | c -s tx |   | x |   | c * x - s * y + tx |
	// | s  c ty | * | y | = | s * x + c * y + ty |
	// | 0  0  1 |   | 1 |   |                  1 |
	map.x = local.x * trans.cos_theta - local.y * trans.sin_theta + trans.trans_x;
	map.y = local.x * trans.sin_theta + local.y * trans.cos_theta + trans.trans_y;
}

const double SQRT_2 = sqrt(2);

inline double calculate_weight_for_obs(const LandmarkObs& obs, 
				       double mu_x, double mu_y, 
				       double sigma_x, double sigma_y) {
	double k = 1 / (2 * M_PI * sigma_x * sigma_y);
	double a = (obs.x - mu_x) / (SQRT_2 * sigma_x);
	double b = (obs.y - mu_y) / (SQRT_2 * sigma_y);
	return k * exp(-(a * a + b * b));
}

ostream& operator << (ostream& s, const Particle& particle) {
	ios init(0);
	init.copyfmt(s);
	s << "ID - ";
	s.width(4);
	s << particle.id
	  << ": ( ";
	s.width(8);
	s << left << particle.x << ", ";
	s.width(8);
	s << left << particle.y << ")  ^ ";
	s.width(8);
	s << left << particle.theta;
	s.copyfmt(init);
	return s;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 1000;
	// sampling particles around the location GPS provided
	
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p.id = i;
		p.x = dist_x(r);
		p.y = dist_y(r);
		p.theta = normalize_angle(dist_theta(r));
		particles.push_back(p);
		// cout<<p<<endl;
	}


	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	for (int i = 0; i < num_particles; ++i) {
		Particle& p = particles[i];

		double dx;
		double dy;
		double dyaw;
		// different equations for different yaw_rate
		if (fabs(yaw_rate) < 1e-5) {
			// for yaw_rate approx = 0
			dx = cos(p.theta) * velocity * delta_t;
			dy = sin(p.theta) * velocity * delta_t;
			// yaw doesn't change
			dyaw = 0;
		} else {
			// for yaw_rate > 0
			double k = velocity / yaw_rate;
			double yaw_kp1 = p.theta + yaw_rate * delta_t;
			dx = k * (sin(yaw_kp1) - sin(p.theta));
			dy = k * (-cos(yaw_kp1) + cos(p.theta));
			dyaw = yaw_rate * delta_t;
		}

		p.x += dx;
		p.y += dy;
		p.theta += dyaw;

		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		// with noise
		p.x = dist_x(r);
		p.y = dist_y(r);
		p.theta = dist_theta(r);
		
		// normalizing yaw
		p.theta = normalize_angle(p.theta);
	}
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	double minDist = std::numeric_limits<double>::max();
	for (int i = 0; i < observations.size(); ++i) {
		LandmarkObs& obs = observations[i];
		for (int j = 0; j < predicted.size(); ++j) {
			const LandmarkObs& prd = predicted[j];
			double d = dist(obs.x, obs.y, prd.x, prd.y);
			if (d < minDist) {
				obs.id = j;
				minDist = d;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double weight_sum = 0.0;
	for (int i = 0; i < num_particles; ++i) {
		Particle& p = particles[i];
		_Transform trans;
		trans.cos_theta = cos(-p.theta);
		trans.sin_theta = sin(-p.theta);
		trans.trans_x = -p.x;
		trans.trans_y = -p.y;
		// calculate predicted location of each landmark
		std::vector<LandmarkObs> predicted;
		predicted.reserve(map_landmarks.landmark_list.size());
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			const Map::single_landmark_s& landmark = map_landmarks.landmark_list[j];
			double d = dist(p.x, p.y,
					landmark.x_f,
					landmark.y_f);
			// if (d > sensor_range) {
			// 	continue;
			// }
			LandmarkObs obs;
			obs.id = landmark.id_i;

			map_coord_to_vehicle_coord(trans, landmark, obs);
			// cout<<obs.id<<"  "<<landmark.x_f<<' '<<landmark.y_f<<" -- "<<p.x<<", "<<p.y<<", "<<p.theta<<" -> "<<obs.x<<", "<<obs.y<<endl;
			predicted.push_back(obs);
		}
		this->dataAssociation(predicted, observations);

		// for debugging: parameters for calling SetAssociations
		// vector<int> associations;
		// vector<double> sense_x;
		// vector<double> sense_y;

		// update weights
		double new_weight = 1.0;
		for (int j = 0; j < observations.size(); ++j) {
			const LandmarkObs& obs = observations[j];
			const LandmarkObs& prd = predicted[obs.id];
			double prob = calculate_weight_for_obs(obs,
							       prd.x,
							       prd.y, 
							       std_landmark[0],
							       std_landmark[1]);
			// cout<<prd.id<<","<<obs.id<<"   "<<prd.x<<' '<<prd.y<<' '<<obs.x<<' '<<obs.y<<endl;
			new_weight *= prob;


			// LandmarkObs global;
			// _Transform trans2;
			// trans2.cos_theta = cos(p.theta);
			// trans2.sin_theta = sin(p.theta);
			// trans2.trans_x = p.x;
			// trans2.trans_y = p.y;

			// vehicle_coord_to_map_coord(trans2, obs, global);

			// associations.push_back(obs.id);
			// sense_x.push_back(global.x);
			// sense_y.push_back(global.y);
		}
		p.weight = new_weight;

		// this->SetAssociations(p, associations, sense_x, sense_y);
		
		weight_sum += p.weight;
	}

	// normalizing
	// for (int i = 0; i < num_particles; ++i) {
	// 	particles[i].weight /= weight_sum;
	// }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<double> weights;
	for (int i = 0; i < num_particles; ++i) {
		weights.push_back(particles[i].weight);
	}
	discrete_distribution<> dist(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; ++i) {
		int idx = dist(r);
		Particle p = particles[idx];
		particles.push_back(p);
	}
	// remove old particles
	particles.erase(particles.begin(), particles.begin() + num_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
