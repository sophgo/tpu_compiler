/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef LAYERGROUP_CYCLE_H
#define LAYERGROUP_CYCLE_H

// copy from profiling/framework/include/common.h
// 1880v2 setting
#define DATA_MAX_DISTANCE 16
#define MAX_BURST_LENGTH 16
#define SPECIAL_FUNCTION_BURST_LENGTH 8
#define AXI_BUS_WIDTH 16 // 16Byte
#define LOCAL_MEM_WIDTH 16 // 16Byte
#define FOUR_KB 0x1000
#define FOUR_KB_MASK 0xfffff000
#define MAX_PACKET_CAPACITY MAX_BURST_LENGTH * AXI_BUS_WIDTH
#define SRC_BASE_ADDR_HIGH_SHIFT 32
#define BYTE64 0x40
#define BYTE64_MASK 0xffffffc0

#define STORE_DATA_MAX_DISTANCE 0
namespace mlir {

static inline uint64_t align_up(uint64_t x, uint64_t n){
  return ceiling_func(x, n) * n;
}

static inline uint64_t align_down(uint64_t x, uint64_t n) {
  return x / n * n;
}

struct Tuple4D {
    union {
        uint64_t batch;
        uint64_t n;
        uint64_t N;
    };

    union {
        uint64_t h;
        uint64_t H;
        uint64_t height;
        uint64_t y;
        uint64_t Y;
    };

    union {
        uint64_t w;
        uint64_t W;
        uint64_t width;
        uint64_t x;
        uint64_t X;
    };

    union {
        uint64_t c;
        uint64_t C;
        uint64_t channel;
        uint64_t z;
        uint64_t Z;
    };

    Tuple4D(uint64_t _n, uint64_t _h, uint64_t _w, uint64_t _c) : N(_n), H(_h), W(_w), C(_c)
    {
    }

    Tuple4D() : N(1), H(0), W(0), C(0)
    {
    }

    inline void reset() {
        this->n = 0;
        this->h = 0;
        this->w = 0;
        this->c = 0;
    }

    inline uint64_t size() {
        return this->n * this->h * this->w  * this->c;
    }

    inline Tuple4D& operator=(const Tuple4D &rOperand) {
        this->n = rOperand.n;
        this->h = rOperand.h;
        this->w = rOperand.w;
        this->c = rOperand.c;
        return *this;
    }

    inline Tuple4D operator+(const Tuple4D &rOperand) {
        Tuple4D ret;
        ret.n = this->n + rOperand.n;
        ret.h = this->h + rOperand.h;
        ret.w = this->w + rOperand.w;
        ret.c = this->c + rOperand.c;
        return ret;
    }

    inline Tuple4D& operator+=(const Tuple4D &rOperand) {
        this->n = this->n + rOperand.n;
        this->h = this->h + rOperand.h;
        this->w = this->w + rOperand.w;
        this->c = this->c + rOperand.c;
        return *this;
    }

    inline Tuple4D operator-(const Tuple4D &rOperand) {
        Tuple4D ret;
        ret.n = this->n - rOperand.n;
        ret.h = this->h - rOperand.h;
        ret.w = this->w - rOperand.w;
        ret.c = this->c - rOperand.c;
        return ret;
    }

    inline Tuple4D& operator-=(const Tuple4D &rOperand) {
        this->n = this->n - rOperand.n;
        this->h = this->h - rOperand.h;
        this->w = this->w - rOperand.w;
        this->c = this->c - rOperand.c;
        return *this;
    }

    inline Tuple4D operator*(const Tuple4D &rOperand) {
        Tuple4D ret;
        ret.n = this->n * rOperand.n;
        ret.h = this->h * rOperand.h;
        ret.w = this->w * rOperand.w;
        ret.c = this->c * rOperand.c;
        return ret;
    }

    inline Tuple4D& operator*=(const Tuple4D &rOperand) {
        this->n = this->n * rOperand.n;
        this->h = this->h * rOperand.h;
        this->w = this->w * rOperand.w;
        this->c = this->c * rOperand.c;
        return *this;
    }

    inline Tuple4D operator/(const Tuple4D &rOperand) {
        Tuple4D ret;
        ret.n = ceil((float)this->n/(float)rOperand.n);
        ret.h = ceil((float)this->h/(float)rOperand.h);
        ret.w = ceil((float)this->w/(float)rOperand.w);
        ret.c = ceil((float)this->c/(float)rOperand.c);
        return ret;
    }

    inline Tuple4D& operator/=(const Tuple4D &rOperand) {
        this->n = ceil((float)this->n/(float)rOperand.n);
        this->h = ceil((float)this->h/(float)rOperand.h);
        this->w = ceil((float)this->w/(float)rOperand.w);
        this->c = ceil((float)this->c/(float)rOperand.c);
        return *this;
    }

    inline bool operator==(const Tuple4D &rOperand) {
        return (this->n == rOperand.n)&&
                (this->h == rOperand.h)&&
                (this->w == rOperand.w)&&
                (this->c == rOperand.c);
    }

    inline bool operator!=(const Tuple4D &rOperand) {
        return !((this->n == rOperand.n)&&
                (this->h == rOperand.h)&&
                (this->w == rOperand.w)&&
                (this->c == rOperand.c));
    }

    inline bool operator<(const Tuple4D &rOperand) {
        return ((this->n < rOperand.n)&&
                (this->h < rOperand.h)&&
                (this->w < rOperand.w)&&
                (this->c < rOperand.c));
    }

    inline bool operator<=(const Tuple4D &rOperand) {
        return ((this->n <= rOperand.n)&&
                (this->h <= rOperand.h)&&
                (this->w <= rOperand.w)&&
                (this->c <= rOperand.c));
    }

    friend std::ostream& operator<<(std::ostream& out, const Tuple4D &operand) {
        out << "(N = " << operand.n << ", H = " << operand.h << ", W = " << operand.w << ", C = " << operand.c << ")";
        return out;
    }
};

}
#endif