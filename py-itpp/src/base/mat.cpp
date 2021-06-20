//! -------------------------------------------------------------------------
//!
//! Copyright (C) 2016 CC0 1.0 Universal (CC0 1.0)
//!
//! The person who associated a work with this deed has dedicated the work to
//! the public domain by waiving all of his or her rights to the work
//! worldwide under copyright law, including all related and neighboring
//! rights, to the extent allowed by law.
//!
//! You can copy, modify, distribute and perform the work, even for commercial
//! purposes, all without asking permission.
//!
//! See the complete legal text at
//! <https://creativecommons.org/publicdomain/zero/1.0/legalcode>
//!
//! -------------------------------------------------------------------------

#include "mat.h"

//! Create wrappers within the mat module
PYBIND11_MODULE(mat, m)
{
 // Default Matrix Type
  generate_itpp_mat_wrapper<double>(m, "mat");

  // Default Complex Matrix Type
  generate_itpp_mat_wrapper<std::complex<double> >(m, "cmat");

  // Default Float Matrix Type
  generate_itpp_mat_wrapper<float>(m, "fmat");

  // Default Complex Float Matrix Type
  generate_itpp_mat_wrapper<std::complex<float> >(m, "cfmat");

  // Integer matrix
  generate_itpp_mat_wrapper<int>(m, "imat");

  // short int matrix
  generate_itpp_mat_wrapper<short int>(m, "smat");

  // bin matrix
  generate_itpp_mat_wrapper<itpp::bin>(m, "bmat");

}
