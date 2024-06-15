"exahype.kernel"() ({
  "exahype.stencil"() ({
    "exahype.patch"() {"patch_name" = "Qcopy", "shape" = [4 : index, 4 : index], "element_type" = 0.000000e+00 : f64} : () -> ()
    "exahype.flux"() ({
      "exahype.call_expr"() ({
      ^0:
      }) {"func" = "Flux_x", "type" = !exahype.empty, "intrinsic" = #exahype.bool"False"} : () -> ()
    }) {"flux_name" = "flux_x", "shape" = [4 : index, 4 : index], "element_type" = 0.000000e+00 : f64, "halo" = [1 : index, 0 : index, 0 : index]} : () -> ()
    "exahype.flux"() ({
      "exahype.call_expr"() ({
      ^1:
      }) {"func" = "Flux_y", "type" = !exahype.empty, "intrinsic" = #exahype.bool"False"} : () -> ()
    }) {"flux_name" = "flux_y", "shape" = [4 : index, 4 : index], "element_type" = 0.000000e+00 : f64, "halo" = [0 : index, 1 : index, 0 : index]} : () -> ()
  }) {"stencil" = ["0[010],0[0-10]", "1[001],1[00-1]"], "scales" = []} : () -> ()
  "exahype.stencil"() ({
    "exahype.patch"() {"patch_name" = "Qcopy", "shape" = [4 : index, 4 : index], "element_type" = 0.000000e+00 : f64} : () -> ()
    "exahype.flux"() ({
      "exahype.call_expr"() ({
      ^2:
      }) {"func" = "X_max_eigenvalues", "type" = !exahype.empty, "intrinsic" = #exahype.bool"False"} : () -> ()
    }) {"flux_name" = "tmp_x_eigen", "shape" = [4 : index, 4 : index], "element_type" = 0.000000e+00 : f64, "halo" = [1 : index, 0 : index, 0 : index]} : () -> ()
    "exahype.flux"() ({
      "exahype.call_expr"() ({
      ^3:
      }) {"func" = "Y_max_eigenvalues", "type" = !exahype.empty, "intrinsic" = #exahype.bool"False"} : () -> ()
    }) {"flux_name" = "tmp_y_eigen", "shape" = [4 : index, 4 : index], "element_type" = 0.000000e+00 : f64, "halo" = [0 : index, 1 : index, 0 : index]} : () -> ()
  }) {"stencil" = ["0[010],0[0-10]", "[1[001],1[00-1]"], "scales" = []} : () -> ()
}) : () -> ()
