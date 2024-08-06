// #define Dimensions 2
extern const int Dimensions = 2;

#include "exahype2/enumerator/AoSLexicographicEnumerator.h"
#include "exahype2/CellData.h"
#include "exahype2/fv/PatchUtils.h"
#include "peano4/utils/Loop.h"
// #include "exahype2/fv/rusanov/rusanov.h"

#include "test.h"
#include <cmath>
#include <iostream>

class MyBaseSolver {
  public:
     virtual void sourceTerm(
      [[maybe_unused]] const double* __restrict__                   Q,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& x,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& h,
      [[maybe_unused]] double                                       t,
      [[maybe_unused]] double                                       dt,
      [[maybe_unused]] double* __restrict__                         S
  );
     virtual double maxEigenvalue(
      [[maybe_unused]] const double* __restrict__                   Q,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& x,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& h,
      [[maybe_unused]] double                                       t,
      [[maybe_unused]] double                                       dt,
      [[maybe_unused]] double* __restrict__                         S
  );
     virtual void flux(
      [[maybe_unused]] const double* __restrict__                   Q,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& x,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& h,
      [[maybe_unused]] double                                       t,
      [[maybe_unused]] double                                       dt,
      [[maybe_unused]] double* __restrict__                         S,
      [[maybe_unused]] double* __restrict__                         F
  );
};

class MySolver : public MyBaseSolver {
 public:
  virtual void sourceTerm(
      [[maybe_unused]] const double* __restrict__                   Q,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& x,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& h,
      [[maybe_unused]] double                                       t,
      [[maybe_unused]] double                                       dt,
      [[maybe_unused]] double* __restrict__                         S
  ) override;
  static void sourceTerm(
      [[maybe_unused]] const double* __restrict__                   Q,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& x,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& h,
      [[maybe_unused]] double                                       t,
      [[maybe_unused]] double                                       dt,
      [[maybe_unused]] double* __restrict__                         S,
      exahype2::Solver::Offloadable // Don't forget this last argument
  );

  virtual double maxEigenvalue(
      [[maybe_unused]] const double* __restrict__                   Q,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& x,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& h,
      [[maybe_unused]] double                                       t,
      [[maybe_unused]] double                                       dt,
      [[maybe_unused]] double* __restrict__                         S
  ) override;
  static double maxEigenvalue(
      [[maybe_unused]] const double* __restrict__                   Q,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& x,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& h,
      [[maybe_unused]] double                                       t,
      [[maybe_unused]] double                                       dt,
      [[maybe_unused]] int                                          normal,
      exahype2::Solver::Offloadable
  );

  virtual void flux(
      [[maybe_unused]] const double* __restrict__                   Q,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& x,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& h,
      [[maybe_unused]] double                                       t,
      [[maybe_unused]] double                                       dt,
      [[maybe_unused]] double* __restrict__                         S,
      [[maybe_unused]] double* __restrict__                         F
  ) override;
  static void flux(
      [[maybe_unused]] const double* __restrict__                   Q,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& x,
      [[maybe_unused]] const tarch::la::Vector<Dimensions, double>& h,
      [[maybe_unused]] double                                       t,
      [[maybe_unused]] double                                       dt,
      [[maybe_unused]] int                                          normal,
      [[maybe_unused]] double* __restrict__                         F,
      exahype2::Solver::Offloadable
  );
};

void initInputData(double* Q, int NumberOfInputEntries) {
  for (int i = 0; i < NumberOfInputEntries; i++) {
    Q[i] = std::sin(3.141 * i / (NumberOfInputEntries));
  }
}

void show_Q(double* Q, int total, int stride_a, int stride_b) {
    for (int i = 0; i < total; i+=stride_a) {
        if (i%(stride_a*stride_b) == 0) {
            std::cout << '\n';
        }
        std::cout << std::ceil(Q[i] * 100.0)/100.0 << '\t'; 
    }
    std::cout << '\n';
}

void old_time_step(double* Q, double dt, const int patch_size, const int halo_size, const int n_real, const int n_aux,double* Qout) {
    int patch_len = patch_size + halo_size;
    auto loopParallelism =  peano4::utils::LoopPlacement();
    auto QInEnum =          exahype2::enumerator::AoSLexicographicEnumerator(1,patch_size,halo_size,n_real,n_aux);
    auto QOutEnum =         exahype2::enumerator::AoSLexicographicEnumerator(1,patch_size,halo_size,n_real,n_aux);
    auto SomeEnum1 =        exahype2::enumerator::AoSLexicographicEnumerator(1,patch_size,halo_size,n_real,n_aux);
    auto SomeEnum2 =        exahype2::enumerator::AoSLexicographicEnumerator(1,patch_size,halo_size,n_real,n_aux);
    auto SomeEnum3 =        exahype2::enumerator::AoSLexicographicEnumerator(1,patch_size,halo_size,n_real,n_aux);
    double *tmp_flux_x =    new double[1*patch_len*patch_len*n_real];
	double *tmp_flux_y =    new double[1*patch_len*patch_len*n_real];
    double *tmp_flux_z =    new double[1*patch_len*patch_len*n_real];
	double *tmp_eigen_x =   new double[1*patch_len*patch_len];
	double *tmp_eigen_y =   new double[1*patch_len*patch_len];
    double *tmp_eigen_z =   new double[1*patch_len*patch_len];
    double *naX =           new double[1*patch_len*patch_len*n_real];
    double *naY =           new double[1*patch_len*patch_len*n_real];
    double *naZ =           new double[1*patch_len*patch_len*n_real];

    const tarch::la::Vector<Dimensions, double > cellCentre{1.0,1.0};
    const tarch::la::Vector<Dimensions, double > cellSize{1.0,1.0};
    // cellCentre_.Vector();
    // cellSize_.Vector();

    double t = 0;
    auto patchData = exahype2::CellData(Q,cellCentre,cellSize,t,dt,Qout);
    // auto patchData = exahype2::CellData(2);

    // exahype2::fv::rusanov::internal::timeStepWithRusanovBatchedStateless<
    // MySolver,
    // patch_size,
    // halo_size/2,
    // n_real,
    // n_aux,
    // 0,   //Flux
    // 0,   //ncp
    // 0,   //Source
    // 0,   //Eigen
    // exahype2::enumerator::AoSLexicographicEnumerator>
    // (PatchData,
    // loopParallelism,
    // QInEnum,
    // QOutEnum,
    // SomeEnum1,SomeEnum2,SomeEnum3,
    // tmp_flux_x,tmp_flux_y,tmp_flux_z,
    // naX,naY,naZ,
    // tmp_eigen_x,tmp_eigen_y,tmp_eigen_z);

    delete[] tmp_eigen_x;
    delete[] tmp_eigen_y;
    delete[] tmp_eigen_z;
    delete[] tmp_flux_x;
    delete[] tmp_flux_y;
    delete[] tmp_flux_z;
    delete[] naX;
    delete[] naY;
    delete[] naZ;
}

int main(int argc, char* argv[]) {
    const int dim = 2;
    const int patch_size = 4;
    const int halo_size = 2*1; //note it is double in value (just to avoid extra multiplications later)
    const int n_real = 5;
    const int n_aux = 5;
    int no_inputs = (n_real+n_aux);//*(patch_size+halo_size)*(patch_size+halo_size)*(patch_size+halo_size);
    for (int i = 0; i < dim; i++) {no_inputs *= (patch_size+halo_size);}

    // allocate memory for input patch
    double* Q1 = new double[no_inputs];
    double* Q2 = new double[no_inputs];
    double* Qout = new double[no_inputs];

    // initialise the inputs
    initInputData(Q1,no_inputs);
    initInputData(Q2,no_inputs);
    
    // do a timestep for both
    time_step(Q1,1);
    old_time_step(Q2,1,patch_size,halo_size,n_real,n_aux,Qout);

    // compare both outputs
    int bad = 0;
    for (int i = 0; i < no_inputs; i++){
        if (Q1[i] != Q2[i]) {
            bad++;
        }
    }
    
    if (bad > 0) {
        std::cout << "there are " << bad << " differences between the two outputs\n";
    }
    else {
        std::cout << "no differences! :)\n";
    }

    // print one of the arrays
    show_Q(Q1,no_inputs,(n_real+n_aux),(patch_size+halo_size));
    show_Q(Q2,no_inputs,(n_real+n_aux),(patch_size+halo_size));

    delete[] Q1;
    delete[] Q2;
    delete[] Qout;
    return 0;
}