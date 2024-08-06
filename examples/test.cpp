#include "Functions.h"

void time_step(double* Q, double dt) {
	double *Q_copy = new double[1*6*6*10];
	double *tmp_flux_x = new double[1*6*6*5];
	double *tmp_flux_y = new double[1*6*6*5];
	double *tmp_eigen_x = new double[1*6*6];
	double *tmp_eigen_y = new double[1*6*6];
	double normal;

	for (int patch = 0; patch < 1; patch++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				for (int var = 0; var < 10; var++) {
					Q_copy[360*patch + 60*i + 10*j + var] = Q[360*patch + 60*i + 10*j + var];
				}
			}
		}
	}
	normal = 0;
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 0; j < 6; j++) {
				for (int var = 0; var < 1; var++) {
					Flux(&Q_copy[360*patch + 60*i + 10*j + var], normal, &tmp_flux_x[180*patch + 30*i + 5*j + var]);
				}
			}
		}
	}
	normal = 1;
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 1; j < 5; j++) {
				for (int var = 0; var < 1; var++) {
					Flux(&Q_copy[360*patch + 60*i + 10*j + var], normal, &tmp_flux_y[180*patch + 30*i + 5*j + var]);
				}
			}
		}
	}
	normal = 0;
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 0; j < 6; j++) {
				for (int var = 0; var < 1; var++) {
					tmp_eigen_x[36*patch + 6*i + 1*j] = maxEigenvalue(&Q_copy[360*patch + 60*i + 10*j + var], normal);
				}
			}
		}
	}
	normal = 1;
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 1; j < 5; j++) {
				for (int var = 0; var < 1; var++) {
					tmp_eigen_y[36*patch + 6*i + 1*j] = maxEigenvalue(&Q_copy[360*patch + 60*i + 10*j + var], normal);
				}
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 0; j < 6; j++) {
				for (int var = 0; var < 5; var++) {
					Q_copy[360*patch + 60*i + 10*j + var] = Q_copy[360*patch + 60*i + 10*j + var] - 0.5*tmp_flux_x[180*patch + 30*(i + 1) + 5*j + var] + 0.5*tmp_flux_x[180*patch + 30*(i - 1) + 5*j + var];
				}
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 1; j < 5; j++) {
				for (int var = 0; var < 5; var++) {
					Q_copy[360*patch + 60*i + 10*j + var] = Q_copy[360*patch + 60*i + 10*j + var] - 0.5*tmp_flux_y[180*patch + 30*i + 5*(j + 1) + var] + 0.5*tmp_flux_y[180*patch + 30*i + 5*(j - 1) + var];
				}
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 0; j < 6; j++) {
				for (int var = 0; var < 1; var++) {
					Q_copy[360*patch + 60*i + 10*j + var] = 0.5*dt*((-Q[360*patch + 60*(i + 1) + 10*j + var] + Q[360*patch + 60*i + 10*j + var])*max(&tmp_eigen_x[36*patch + 6*(i + 1) + 1*j], &tmp_eigen_x[36*patch + 6*i + 1*j]) + (Q[360*patch + 60*(i - 1) + 10*j + var] - Q[360*patch + 60*i + 10*j + var])*max(&tmp_eigen_x[36*patch + 6*(i - 1) + 1*j], &tmp_eigen_x[36*patch + 6*i + 1*j])) + Q_copy[360*patch + 60*i + 10*j + var];
				}
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 1; j < 5; j++) {
				for (int var = 0; var < 1; var++) {
					Q_copy[360*patch + 60*i + 10*j + var] = 0.5*dt*((-Q[360*patch + 60*i + 10*(j + 1) + var] + Q[360*patch + 60*i + 10*j + var])*max(&tmp_eigen_y[36*patch + 6*i + 1*(j + 1)], &tmp_eigen_y[36*patch + 6*i + 1*j]) + (Q[360*patch + 60*i + 10*(j - 1) + var] - Q[360*patch + 60*i + 10*j + var])*max(&tmp_eigen_y[36*patch + 6*i + 1*(j - 1)], &tmp_eigen_y[36*patch + 6*i + 1*j])) + Q_copy[360*patch + 60*i + 10*j + var];
				}
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 1; j < 5; j++) {
				for (int var = 0; var < 10; var++) {
					Q[360*patch + 60*i + 10*j + var] = Q_copy[360*patch + 60*i + 10*j + var];
				}
			}
		}
	}

	delete[] Q_copy;
	delete[] tmp_flux_x;
	delete[] tmp_flux_y;
	delete[] tmp_eigen_x;
	delete[] tmp_eigen_y;
}
