/*
Copyright 2020 Alexander Brauckmann

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <memory>
#include <tuple>
#include <vector>
#include <map>

#include "common/clang_driver.h"
#include "common/llvm_driver.h"
#include "common/visitor.h"

namespace compy {
namespace llvm {

namespace seq {
struct InstructionInfo;
using InstructionInfoPtr = std::shared_ptr<InstructionInfo>;

struct BasicBlockInfo;
using BasicBlockInfoPtr = std::shared_ptr<BasicBlockInfo>;

struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct InstructionInfo : IVisitee {
  std::vector<std::string> tokens;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct BasicBlockInfo : IVisitee {
  std::string name;
  std::vector<InstructionInfoPtr> instructions;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : instructions) it->accept(v);
  }
};

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;
  std::vector<BasicBlockInfoPtr> basicBlocks;
  std::string str;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : basicBlocks) it->accept(v);
  }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : functionInfos) it->accept(v);
  }
};
}  // namespace seq

namespace graph {
struct OperandInfo;
using OperandInfoPtr = std::shared_ptr<OperandInfo>;

struct ArgInfo;
using ArgInfoPtr = std::shared_ptr<ArgInfo>;

struct ConstantInfo;
using ConstantInfoPtr = std::shared_ptr<ConstantInfo>;

struct InstructionInfo;
using InstructionInfoPtr = std::shared_ptr<InstructionInfo>;

struct BasicBlockInfo;
using BasicBlockInfoPtr = std::shared_ptr<BasicBlockInfo>;

struct MemoryAccessInfo;
using MemoryAccessInfoPtr = std::shared_ptr<MemoryAccessInfo>;

struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct CallGraphInfo;
using CallGraphInfoPtr = std::shared_ptr<CallGraphInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct OperandInfo : IVisitee {
  virtual ~OperandInfo() = default;
};

struct ArgInfo : OperandInfo {
  std::string name;
  std::string type;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ConstantInfo : OperandInfo {
    std::string type;
    std::string value;

    void accept(IVisitor* v) override { v->visit(this); }
};

struct InstructionInfo : OperandInfo {
  std::string instStr;
  std::string type;
  std::string opcode;
  std::string callTarget;
  bool isLoadOrStore;
  std::vector<OperandInfoPtr> operands;
  FunctionInfoPtr function;
  BasicBlockInfoPtr basicBlock;
  double recipThroughput; // TCK_RecipThroughput
  double latency; // TCK_Latency
  double codeSize; // TCK_CodeSize
  void accept(IVisitor* v) override { v->visit(this); }
};

struct BasicBlockInfo : IVisitee {
  std::string name;
  std::string fullName;
  std::vector<InstructionInfoPtr> instructions;
  std::vector<BasicBlockInfoPtr> successors;
  double frequency;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : instructions) it->accept(v);
  }
};

struct MemoryAccessInfo {
  std::string type;
  InstructionInfoPtr inst;
  BasicBlockInfoPtr basicBlock;
  std::vector<MemoryAccessInfoPtr> dependencies;
};

struct FunctionInfo : IVisitee {
  std::string name;
  std::string type;
  InstructionInfoPtr entryInstruction;
  std::vector<InstructionInfoPtr> exitInstructions;
  std::vector<ArgInfoPtr> args;
  std::vector<BasicBlockInfoPtr> basicBlocks;
  std::vector<MemoryAccessInfoPtr> memoryAccesses;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : basicBlocks) it->accept(v);
  }
};

struct CallGraphInfo {
  std::vector<std::string> calls;
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;
  CallGraphInfoPtr callGraphInfo;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : functionInfos) it->accept(v);
  }
};
}  // namespace graph

namespace names {
struct GlobalInfo;
using GlobalInfoPtr = std::shared_ptr<GlobalInfo>;

struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct GlobalInfo : IVisitee {
  std::string name;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;

  void accept(IVisitor* v) override { v->visit(this); }

};

struct ExtractionInfo : IVisitee {
  std::vector<GlobalInfoPtr> globalInfos;
  std::vector<FunctionInfoPtr> functionInfos;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : globalInfos) it->accept(v);
    for (const auto &it : functionInfos) it->accept(v);
  }
};
}  // namespace names

namespace msf {
struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;
  std::map<std::string, float> features;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;
  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : functionInfos) it->accept(v);
  }
};

}  // namespace msf

namespace loop {
struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;
  std::map<std::string, float> features;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;
  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : functionInfos) it->accept(v);
  }
};

}  // namespace loop

namespace insts {
struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;
  int instructions;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;
  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : functionInfos) it->accept(v);
  }
};

}  // namespace insts

namespace histogram {
struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;
  std::map<std::string, float> instructions;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;
  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : functionInfos) it->accept(v);
  }
};

}  // namespace histogram


namespace opcodes {
struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;
  std::vector<std::string> opcodes;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;
  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : functionInfos) it->accept(v);
  }
};

}  // namespace opcodes

namespace wlcost {
struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;
  double recipThroughput; // TCK_RecipThroughput
  double latency; // TCK_Latency
  double codeSize; // TCK_CodeSize
  double oneCost; // 1.0

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;
  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto &it : functionInfos) it->accept(v);
  }
};
}  // namespace wlcost

namespace ir2vec {
struct InstructionInfo;
using InstructionInfoPtr = std::shared_ptr<InstructionInfo>;

struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ModuleInfo;
using ModuleInfoPtr = std::shared_ptr<ModuleInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct InstructionInfo : IVisitee {
  std::string instStr;
  std::string opcode;
  std::vector<double> ir2vec;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;
  std::vector<double> ir2vec;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ModuleInfo : IVisitee {
  std::string name;
  std::vector<double> ir2vec;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;
  std::vector<InstructionInfoPtr> instructionInfos;
  ModuleInfoPtr moduleInfo;

  void accept(IVisitor* v) override {
    v->visit(this);
    moduleInfo->accept(v);
    for (const auto &it : functionInfos) it->accept(v);
    for (const auto &it : instructionInfos) it->accept(v);
  }
};
}  // namespace ir2vec

class LLVMIRExtractor {
 public:
  LLVMIRExtractor(ClangDriverPtr clangDriver);
  LLVMIRExtractor(LLVMDriverPtr llvmDriver);
  graph::ExtractionInfoPtr GraphFromSource(std::string filename);
  graph::ExtractionInfoPtr GraphFromIR(std::string filename);
  seq::ExtractionInfoPtr SeqFromSource(std::string filename);
  seq::ExtractionInfoPtr SeqFromIR(std::string filename);
  msf::ExtractionInfoPtr MSFFromSource(std::string filename);
  msf::ExtractionInfoPtr MSFFromIR(std::string filename);
  loop::ExtractionInfoPtr LoopFromSource(std::string filename);
  loop::ExtractionInfoPtr LoopFromIR(std::string filename);
  names::ExtractionInfoPtr NamesFromSource(std::string filename);
  names::ExtractionInfoPtr NamesFromIR(std::string filename);
  insts::ExtractionInfoPtr InstsFromSource(std::string filename);
  insts::ExtractionInfoPtr InstsFromIR(std::string filename);
  histogram::ExtractionInfoPtr HistogramFromSource(std::string filename);
  histogram::ExtractionInfoPtr HistogramFromIR(std::string filename);
  opcodes::ExtractionInfoPtr OpcodesFromSource(std::string filename);
  opcodes::ExtractionInfoPtr OpcodesFromIR(std::string filename);
  wlcost::ExtractionInfoPtr WLCostFromSource(std::string filename);
  wlcost::ExtractionInfoPtr WLCostFromIR(std::string filename);
  ir2vec::ExtractionInfoPtr IR2VecFromSource(std::string filename);
  ir2vec::ExtractionInfoPtr IR2VecFromIR(std::string filename);

 private:
  ClangDriverPtr clangDriver_;
  LLVMDriverPtr llvmDriver_;
};

}  // namespace llvm
}  // namespace compy
