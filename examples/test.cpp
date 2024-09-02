#include "Functions.h"


#include "exahype2/UserInterface.h"
#include "observers/CreateGrid.h"
#include "observers/CreateGridAndConvergeLoadBalancing.h"
#include "observers/CreateGridButPostponeRefinement.h"
#include "observers/InitGrid.h"
#include "observers/PlotSolution.h"
#include "observers/TimeStep.h"
#include "peano4/peano.h"
#include "repositories/DataRepository.h"
#include "repositories/SolverRepository.h"
#include "repositories/StepRepository.h"
#include "tarch/accelerator/accelerator.h"
#include "tarch/accelerator/Device.h"
#include "tarch/logging/CommandLineLogger.h"
#include "tarch/logging/Log.h"
#include "tarch/logging/LogFilter.h"
#include "tarch/logging/Statistics.h"
#include "tarch/multicore/Core.h"
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/otter.h"
#include "tarch/NonCriticalAssertions.h"
#include "tarch/timing/Measurement.h"
#include "tarch/timing/Watch.h"
#include "tasks/FVRusanovSolverEnclaveTask.h"
#include "toolbox/loadbalancing/loadbalancing.h"


void time_step(double* dim, double patch_size, double halo_size, double n_real, double n_aux, double Q, double dt) {
	double *Q_copy = new double[1*6*6*10];
	double *tmp_flux_x = new double[1*6*6*5];
	double *tmp_flux_y = new double[1*6*6*5];
	double *tmp_eigen_x = new double[1*6*6];
	double *tmp_eigen_y = new double[1*6*6];
	double normal;

	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 1; j < 5; j++) {
				for (int var = 0; var < 10; var++) {
					Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1) + var] = Q[360*patch + 60*i + 10*j + var];
				}
			}
		}
	}
	normal = 0;
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 0; j < 6; j++) {
				Flux(&&Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1)], normal, &tmp_flux_x[180*patch + 30*i + 5*j]);
			}
		}
	}
	normal = 1;
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 1; j < 5; j++) {
				Flux(&&Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1)], normal, &tmp_flux_y[180*patch + 30*i + 5*j]);
			}
		}
	}
	normal = 0;
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 0; j < 6; j++) {
				tmp_eigen_x[36*patch + 6*i + 1*j] = maxEigenvalue(&&Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1)], normal);
			}
		}
	}
	normal = 1;
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 1; j < 5; j++) {
				tmp_eigen_y[36*patch + 6*i + 1*j] = maxEigenvalue(&&Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1)], normal);
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 0; j < 6; j++) {
				for (int var = 0; var < 5; var++) {
					Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1) + var] = Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1) + var] - 0.5*tmp_flux_x[180*patch + 30*(i + 1) + 5*j + var] + 0.5*tmp_flux_x[180*patch + 30*(i - 1) + 5*j + var];
				}
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 1; j < 5; j++) {
				for (int var = 0; var < 5; var++) {
					Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1) + var] = Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1) + var] - 0.5*tmp_flux_y[180*patch + 30*i + 5*(j + 1) + var] + 0.5*tmp_flux_y[180*patch + 30*i + 5*(j - 1) + var];
				}
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 0; j < 6; j++) {
				Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1)] = 0.5*dt*((-Q[360*patch + 60*(i + 1) + 10*j] + Q[360*patch + 60*i + 10*j])*max(&tmp_eigen_x[36*patch + 6*(i + 1) + 1*j], &tmp_eigen_x[36*patch + 6*i + 1*j]) + (Q[360*patch + 60*(i - 1) + 10*j] - Q[360*patch + 60*i + 10*j])*max(&tmp_eigen_x[36*patch + 6*(i - 1) + 1*j], &tmp_eigen_x[36*patch + 6*i + 1*j])) + Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1)];
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 0; i < 6; i++) {
			for (int j = 1; j < 5; j++) {
				Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1)] = 0.5*dt*((-Q[360*patch + 60*i + 10*(j + 1)] + Q[360*patch + 60*i + 10*j])*max(&tmp_eigen_y[36*patch + 6*i + 1*(j + 1)], &tmp_eigen_y[36*patch + 6*i + 1*j]) + (Q[360*patch + 60*i + 10*(j - 1)] - Q[360*patch + 60*i + 10*j])*max(&tmp_eigen_y[36*patch + 6*i + 1*(j - 1)], &tmp_eigen_y[36*patch + 6*i + 1*j])) + Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1)];
			}
		}
	}
	for (int patch = 0; patch < 1; patch++) {
		for (int i = 1; i < 5; i++) {
			for (int j = 1; j < 5; j++) {
				for (int var = 0; var < 10; var++) {
					Q[360*patch + 60*i + 10*j + var] = Q_copy[360*(patch - 1) + 60*(i - 1) + 10*(j - 1) + var];
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
