#ifndef _TPU_BACKENDCONTEXT_H_
#define _TPU_BACKENDCONTEXT_H_

#include <vector>
#include <cstring>
#include <assert.h>

typedef struct {
  int chip_version;
  int nodechip_shift;
  int npu_shift;
  int eu_shift;
  int local_mem_shift;
  int local_mem_banks;
  uint64_t global_mem_size;
  int nodechip_num;
  int npu_num;
  int eu_num;
  int local_mem_size;
  int unit_size;
} hw_info_t;

class BackendContext {

public:
  virtual ~BackendContext() {}

  // compile the graph to cmdbuf.
  //virtual void build(NetParameter &net, const char* term_layer=nullptr) = 0;
  //virtual void enter() = 0;
  //virtual void exit() = 0;
  //virtual void submit() = 0;

  void write_cmdbuf(const void* cmdbuf, uint32_t size) {
    cmdbuf_.resize(size);
    memcpy(&cmdbuf_[0], cmdbuf, size);
  }

  void read_cmdbuf(std::vector<uint8_t>& out_cmdbuf) {
    out_cmdbuf.assign(cmdbuf_.begin(), cmdbuf_.end());
  }

public:
  hw_info_t hw;

protected:
  BackendContext() : cmdbuf_() {}

private:
  std::vector<uint8_t> cmdbuf_;
};


/// Writes BM188x machine code to a stream.
class BM188xBackendContext : public BackendContext {
 public:
  BM188xBackendContext(std::vector<int8_t> &weight)
      : BackendContext() {
    weight_ = weight.data();
    weight_size_ = weight.size() * sizeof(int8_t);
  }

  ~BM188xBackendContext() {};

  virtual void parallel_enable() const = 0;
  virtual void parallel_disable() const = 0;

  virtual void set_layer_id(u16 layer_id) const = 0;
  virtual int layer_id() const =0;

  //void build(NetParameter &net, const char *term_layer = nullptr) override;
  //void enter() override;
  //void exit() override;
  //virtual void submit() const = 0;

  void *weight() { return weight_; }
  uint64_t weight_size() { return weight_size_; }
  //int8_t read_weight(uint64_t offset);

 protected:
  bmk_info_t bmk_info_;
  void *weight_;
  uint64_t weight_size_;
  //void *emiter_;
};

#endif // _TPU_BACKENDCONTEXT_H_
