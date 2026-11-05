// pybind11 module: lets the launch file validate a profile with the same C++ code as the node.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "strataslam/config/config_loader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_strataslam_py, m) {
  m.doc() = "strataslam core bindings";
  m.def(
      "validate_profile",
      [](const std::string& path) {
        const auto r = strataslam::load_profile_file(path);
        // Messages may quote the profile's own bytes (a bad escape such as "\\ñ" in a quoted string), which need not be
        // valid UTF-8: decoded with replacement, so the launch prints the error instead of dying with
        // UnicodeDecodeError (audit A7-02).
        const auto text = [](const std::string& s) {
          return py::reinterpret_steal<py::str>(PyUnicode_DecodeUTF8(s.data(), static_cast<Py_ssize_t>(s.size()), "replace"));
        };
        py::list errors, warnings;
        for (const auto& e : r.errors) errors.append(text(e.format()));
        for (const auto& w : r.warnings) warnings.append(text("config warning: " + (w.path.empty() ? w.message : w.path + ": " + w.message)));
        return py::make_tuple(errors, warnings);
      },
      py::arg("path"), "Returns (errors, warnings) as formatted strings; errors are empty for a valid profile.");
}
