#ifndef CVI_CUSTOM_OP_PARAMETER_H
#define CVI_CUSTOM_OP_PARAMETER_H
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <typeinfo>
#include <memory>
#include <assert.h>

namespace cvi {

class FieldBase {
public:
  FieldBase(const char *signature) : signature(signature) {}
  virtual ~FieldBase() = default;
  const char *signature;
};

template <typename T>
class Field : public FieldBase {
public:
  Field(T &val) : FieldBase(typeid(T).name()), data(val) {}
  T data;
};

class OpParam {
public:
  template <typename T>
  void put(std::string name, T value) {
    fields[name] = std::make_shared<Field<T>>(value);
  }

  template <typename T>
  T &get(std::string name) {
    auto f = dynamic_cast<Field<T> *>(fields[name].get());
    assert(f);
    return f->data;
  }

  bool has(std::string name) {
    auto it = fields.find(name);
    return (it != fields.end());
  }

  std::map<std::string, std::shared_ptr<FieldBase>> fields;
};

}
#endif