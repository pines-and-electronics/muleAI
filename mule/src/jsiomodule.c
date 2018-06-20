#include <Python.h>
#include <linux/joystick.h>

static char module_docstring[] = "This module retrieve hardware joystick constants.";

static PyObject * _jsio_retrieve_JSIOCGAXES(PyObject * self, PyObject * args);
static PyObject * _jsio_retrieve_JSIOCGBUTTONS(PyObject * self, PyObject * args);
static PyObject * _jsio_retrieve_JSIOCGNAME(PyObject * self, PyObject * args);
static PyObject * _jsio_retrieve_JSIOCGAXMAP(PyObject * self, PyObject * args);
static PyObject * _jsio_retrieve_JSIOCGBTNMAP(PyObject * self, PyObject * args);
static PyObject * _jsio_retrieve_JS_EVENT_BUTTON(PyObject * self, PyObject * args);
static PyObject * _jsio_retrieve_JS_EVENT_AXIS(PyObject * self, PyObject * args);
static PyObject * _jsio_retrieve_JS_EVENT_INIT(PyObject * self, PyObject * args);
static PyObject * _jsio_retrieve_MAX_NR_AXES(PyObject * self, PyObject * args);
static PyObject * _jsio_retrieve_MAX_NR_BUTTONS(PyObject * self, PyObject * args);


static PyMethodDef jsio_methods[] = {
	{"retrieve_JSIOCGAXES", _jsio_retrieve_JSIOCGAXES, METH_NOARGS, NULL},
	{"retrieve_JSIOCGBUTTONS", _jsio_retrieve_JSIOCGBUTTONS, METH_NOARGS, NULL},
	{"retrieve_JSIOCGNAME", _jsio_retrieve_JSIOCGNAME, METH_VARARGS, NULL},
	{"retrieve_JSIOCGAXMAP", _jsio_retrieve_JSIOCGAXMAP, METH_NOARGS, NULL},
	{"retrieve_JSIOCGBTNMAP", _jsio_retrieve_JSIOCGBTNMAP, METH_NOARGS, NULL},
	{"retrieve_JS_EVENT_BUTTON", _jsio_retrieve_JS_EVENT_BUTTON, METH_NOARGS, NULL},
	{"retrieve_JS_EVENT_AXIS", _jsio_retrieve_JS_EVENT_AXIS, METH_NOARGS, NULL},
	{"retrieve_JS_EVENT_INIT", _jsio_retrieve_JS_EVENT_INIT, METH_NOARGS, NULL},
	{"retrieve_MAX_NR_AXES", _jsio_retrieve_MAX_NR_AXES, METH_NOARGS, NULL},
	{"retrieve_MAX_NR_BUTTONS", _jsio_retrieve_MAX_NR_BUTTONS, METH_NOARGS, NULL},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef jsio_module = {
	PyModuleDef_HEAD_INIT,
	"_jsio", 
	module_docstring, 
	-1, 
	jsio_methods
};

// This module is called _jsio, so PyInit_<module-name> becomes PYInit__jsio
PyMODINIT_FUNC PyInit__jsio(void) {
	return PyModule_Create(&jsio_module);
}




static PyObject * _jsio_retrieve_JSIOCGAXES(PyObject * self, PyObject * args) {
	PyObject * value = Py_BuildValue("I", JSIOCGAXES);
	return value;
}

static PyObject * _jsio_retrieve_JSIOCGBUTTONS(PyObject * self, PyObject * args) {
	PyObject * value = Py_BuildValue("I", JSIOCGBUTTONS);
	return value;
}

static PyObject * _jsio_retrieve_JSIOCGNAME(PyObject * self, PyObject * args) {
	unsigned int length;
	PyArg_ParseTuple(args, "I", &length);
	PyObject * value = Py_BuildValue("I", JSIOCGNAME(length));
	return value;
}

static PyObject * _jsio_retrieve_JSIOCGAXMAP(PyObject * self, PyObject * args) {
	PyObject * value = Py_BuildValue("I", JSIOCGAXMAP);
	return value;
}

static PyObject * _jsio_retrieve_JSIOCGBTNMAP(PyObject * self, PyObject * args) {
	PyObject * value = Py_BuildValue("I", JSIOCGBTNMAP);
	return value;
}


static PyObject * _jsio_retrieve_JS_EVENT_BUTTON(PyObject * self, PyObject * args) {
	PyObject * value = Py_BuildValue("I", JS_EVENT_BUTTON);
	return value;
}

static PyObject * _jsio_retrieve_JS_EVENT_AXIS(PyObject * self, PyObject * args) {
	PyObject * value = Py_BuildValue("I", JS_EVENT_AXIS);
	return value;
}

static PyObject * _jsio_retrieve_JS_EVENT_INIT(PyObject * self, PyObject * args) {
	PyObject * value = Py_BuildValue("I", JS_EVENT_INIT);
	return value;
}

static PyObject * _jsio_retrieve_MAX_NR_AXES(PyObject * self, PyObject * args) {
	PyObject * value = Py_BuildValue("I", ABS_CNT);
	return value;
}


static PyObject * _jsio_retrieve_MAX_NR_BUTTONS(PyObject * self, PyObject * args) {
	PyObject * value = Py_BuildValue("I", KEY_MAX - BTN_MISC + 1);
	return value;
}

