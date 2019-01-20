
#if !defined(__PY_UTIL__)
#define __PY_UTIL__

#include <Python.h>
#include <unordered_map>
#include <string>
#include <memory>

class PyUtil {
public:
  inline PyUtil() {
    Py_Initialize();

    if (!Py_IsInitialized()) {
      throw std::runtime_error("python init error");
    }
  }

  inline ~PyUtil() { Py_Finalize(); }

  inline static std::shared_ptr<PyObject> get_module(std::string module_name) {
    if (PyUtil::modules.find(module_name) == PyUtil::modules.end()) {
      auto module = PyUtil::create_shared_ptr(
          PyImport_ImportModule(module_name.c_str()));
      if (!module.get()) {
        throw std::runtime_error("import module error");
      }

      PyUtil::modules[module_name] = module;
    }

    return PyUtil::modules[module_name];
  }

  template <class... Args>
  static std::shared_ptr<PyObject>
  create_object(std::shared_ptr<PyObject> module, std::string class_name,
                Args&& ... args) {
    auto dict = PyUtil::create_shared_ptr(PyModule_GetDict(module.get()));
    if (!dict.get()) {
      throw std::runtime_error("get dict error");
    }

    auto clss = PyUtil::create_shared_ptr(
        PyDict_GetItemString(dict.get(), class_name.c_str()));
    if (!clss.get()) {
      throw std::runtime_error("get class error");
    }

    auto ins =
        PyUtil::create_shared_ptr(PyInstance_New(clss.get(), std::forward(args)));
    if (!ins.get()) {
      throw std::runtime_error("create instance error");
    }

    return ins;
  }

  template <class... Args>
  static std::shared_ptr<PyObject>
  call_object_method(std::shared_ptr<PyObject> object, std::string method_name,
                     std::string args_format, Args&& ... args) {
    auto res = PyUtil::create_shared_ptr(PyObject_CallMethod(
        object.get(), method_name.c_str(), args_format.c_str(), std::forward(args)));
    if (!res.get()) {
      throw std::runtime_error("call object method error");
    }

    return res;
  }

  inline static std::shared_ptr<PyObject> create_shared_ptr(PyObject *ptr) {
    return std::shared_ptr<PyObject>(ptr,
                                     [](PyObject *ptr) { Py_DECREF(ptr); });
  }

private:
  static std::unordered_map<std::string, std::shared_ptr<PyObject>> modules;

  static PyUtil instance;
};

#endif // __PY_UTIL__
