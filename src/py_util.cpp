#include <py_util.h>

// init static vairables
std::unordered_map<std::string, std::shared_ptr<PyObject>> PyUtil::modules;

PyUtil PyUtil::instance;
