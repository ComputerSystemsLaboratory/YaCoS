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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "clang_ast/clang_extractor.h"
#include "common/clang_driver.h"
#include "llvm_ir/llvm_extractor.h"

using namespace compy;

namespace py = pybind11;
namespace cg = compy::clang::graph;
namespace cs = compy::clang::seq;
namespace lg = compy::llvm::graph;
namespace ls = compy::llvm::seq;
namespace ln = compy::llvm::names;
namespace lm = compy::llvm::msf;
namespace li = compy::llvm::insts;
namespace lwlc = compy::llvm::wlcost;
namespace li2v = compy::llvm::ir2vec;

using CD = compy::ClangDriver;
using LD = compy::LLVMDriver;
using CE = compy::clang::ClangExtractor;
using LE = compy::llvm::LLVMIRExtractor;

namespace pybind11 {
template <>
struct polymorphic_type_hook<lg::OperandInfo> {
  static const void *get(const lg::OperandInfo *src,
                         const std::type_info *&type) {
    if (src) {
      if (dynamic_cast<const lg::ArgInfo *>(src) != nullptr) {
        type = &typeid(lg::ArgInfo);
        return static_cast<const lg::ArgInfo *>(src);
      }
      if (dynamic_cast<const lg::InstructionInfo *>(src) != nullptr) {
        type = &typeid(lg::InstructionInfo);
        return static_cast<const lg::InstructionInfo *>(src);
      }
      if (dynamic_cast<const lg::ConstantInfo *>(src) != nullptr) {
        type = &typeid(lg::ConstantInfo);
        return static_cast<const lg::ConstantInfo *>(src);
      }
    }
    return src;
  }
};

template <>
struct polymorphic_type_hook<cg::OperandInfo> {
  static const void *get(const cg::OperandInfo *src,
                         const std::type_info *&type) {
    if (src) {
      if (dynamic_cast<const cg::DeclInfo *>(src) != nullptr) {
        type = &typeid(cg::DeclInfo);
        return static_cast<const cg::DeclInfo *>(src);
      }
      if (dynamic_cast<const cg::StmtInfo *>(src) != nullptr) {
        type = &typeid(cg::StmtInfo);
        return static_cast<const cg::StmtInfo *>(src);
      }
    }
    return src;
  }
};
}  // namespace pybind11

class PyVisitor : public IVisitor {
 public:
  using IVisitor::IVisitor;  /* Inherit the constructors */

  void visit(IVisitee* v) override {
    PYBIND11_OVERLOAD_PURE(
        void,          /* Return type */
        IVisitor,      /* Parent class */
        visit,         /* Name of function in C++ (must match Python name) */
        v              /* Argument(s) */
    );
  }
};

void registerClangDriver(py::module m) {
  py::class_<CD, std::shared_ptr<CD>> clangDriver(m, "ClangDriver");
  clangDriver.def(
      py::init<CD::ProgrammingLanguage, CD::OptimizationLevel,
          std::vector<std::tuple<std::string, CD::IncludeDirType>>,
          std::vector<std::string>>())
      .def("addIncludeDir", &CD::addIncludeDir)
      .def("removeIncludeDir", &CD::removeIncludeDir)
      .def("setOptimizationLevel", &CD::setOptimizationLevel);

  py::enum_<CD::ProgrammingLanguage>(clangDriver, "ProgrammingLanguage")
      .value("C", CD::ProgrammingLanguage::C)
      .value("CPlusPlus", CD::ProgrammingLanguage::CPLUSPLUS)
      .value("OpenCL", CD::ProgrammingLanguage::OPENCL)
      .export_values();

  py::enum_<CD::OptimizationLevel>(clangDriver, "OptimizationLevel")
      .value("O0", CD::OptimizationLevel::O0)
      .value("O1", CD::OptimizationLevel::O1)
      .value("O2", CD::OptimizationLevel::O2)
      .value("O3", CD::OptimizationLevel::O3)
      .value("Os", CD::OptimizationLevel::Os)
      .value("Oz", CD::OptimizationLevel::Oz)
      .export_values();

  py::enum_<CD::IncludeDirType>(clangDriver, "IncludeDirType")
      .value("System", CD::IncludeDirType::SYSTEM)
      .value("User", CD::IncludeDirType::USER)
      .export_values();
}

void registerLLVMDriver(py::module m) {
  py::class_<LD, std::shared_ptr<LD>> LLVMDriver(m, "LLVMDriver");
  LLVMDriver.def(py::init<std::vector<std::string>>());
  LLVMDriver.def(py::init<>())
      .def("addOptimizationFront", &LD::addOptimizationFront)
      .def("addOptimizationBack", &LD::addOptimizationBack)
      .def("addOptimizationsFront", &LD::addOptimizationsFront)
      .def("addOptimizationsBack", &LD::addOptimizationsBack)
      .def("removeOptimization", &LD::removeOptimization)
      .def("removeOptimizationsFront", &LD::removeOptimizationsFront)
      .def("removeOptimizationsBack", &LD::removeOptimizationsBack)
      .def("clearOptimizations", &LD::clearOptimizations)
      .def("setOptimizations", &LD::setOptimizations)
      .def("getOptimizations", &LD::getOptimizations);
}

void registerClangExtractor(py::module m_parent) {
  // Extractor
  py::class_<CE> clangExtractor(m_parent, "ClangExtractor");
  clangExtractor.def(py::init<ClangDriverPtr>());
  clangExtractor.def("GraphFromSource", &CE::GraphFromSource);
  clangExtractor.def("SeqFromSource", &CE::SeqFromSource);

  py::module m = m_parent.def_submodule("clang");

  // Subtypes
  py::module m_graph = m.def_submodule("graph");

  // Graph extractor
  py::class_<cg::ExtractionInfo, std::shared_ptr<cg::ExtractionInfo>>(
      m_graph, "ExtractionInfo")
      .def("accept", &cg::ExtractionInfo::accept)
      .def_readonly("functionInfos", &cg::ExtractionInfo::functionInfos);

  py::class_<cg::DeclInfo, std::shared_ptr<cg::DeclInfo>>(m_graph,
                                                          "DeclInfo")
      .def_readonly("name", &cg::DeclInfo::name)
      .def_readonly("type", &cg::DeclInfo::type);

  py::class_<cg::FunctionInfo, std::shared_ptr<cg::FunctionInfo>>(
      m_graph, "FunctionInfo")
      .def("accept", &cg::FunctionInfo::accept)
      .def_readonly("name", &cg::FunctionInfo::name)
      .def_readonly("type", &cg::FunctionInfo::type)
      .def_readonly("args", &cg::FunctionInfo::args)
      .def_readonly("cfgBlocks", &cg::FunctionInfo::cfgBlocks)
      .def_readonly("entryStmt", &cg::FunctionInfo::entryStmt);

    py::class_<cg::CFGBlockInfo, std::shared_ptr<cg::CFGBlockInfo>>(
            m_graph, "CFGBlockInfo")
            .def_readonly("name", &cg::CFGBlockInfo::name)
            .def_readonly("statements", &cg::CFGBlockInfo::statements)
            .def_readonly("successors", &cg::CFGBlockInfo::successors);

  py::class_<cg::StmtInfo, std::shared_ptr<cg::StmtInfo>>(m_graph,
                                                          "StmtInfo")
      .def_readonly("name", &cg::StmtInfo::name)
      .def_readonly("operation", &cg::StmtInfo::operation)
      .def_readonly("ast_relations", &cg::StmtInfo::ast_relations)
      .def_readonly("ref_relations", &cg::StmtInfo::ref_relations);

  // Sequence extractor
  py::module m_seq = m.def_submodule("seq");

  py::class_<cs::ExtractionInfo, std::shared_ptr<cs::ExtractionInfo>>(
      m_seq, "ExtractionInfo")
      .def("accept", &cs::ExtractionInfo::accept)
      .def_readonly("functionInfos", &cs::ExtractionInfo::functionInfos);

  py::class_<cs::FunctionInfo, std::shared_ptr<cs::FunctionInfo>>(
      m_seq, "FunctionInfo")
      .def("accept", &cs::FunctionInfo::accept)
      .def_readonly("name", &cs::FunctionInfo::name)
      .def_readonly("tokenInfos", &cs::FunctionInfo::tokenInfos);

  py::class_<cs::TokenInfo, std::shared_ptr<cs::TokenInfo>>(
      m_seq, "TokenInfo")
      .def_readonly("name", &cs::TokenInfo::name)
      .def_readonly("kind", &cs::TokenInfo::kind);
}

void registerLLVMExtractor(py::module m_parent) {
  // Extractor
  py::class_<LE> llvmExtractor(m_parent, "LLVMIRExtractor");
  llvmExtractor.def(py::init<ClangDriverPtr>());
  llvmExtractor.def(py::init<LLVMDriverPtr>());
  llvmExtractor.def("GraphFromSource", &LE::GraphFromSource);
  llvmExtractor.def("GraphFromIR", &LE::GraphFromIR);
  llvmExtractor.def("SeqFromSource", &LE::SeqFromSource);
  llvmExtractor.def("SeqFromIR", &LE::SeqFromIR);
  llvmExtractor.def("MSFFromSource", &LE::MSFFromSource);
  llvmExtractor.def("MSFFromIR", &LE::MSFFromIR);
  llvmExtractor.def("NamesFromSource", &LE::NamesFromSource);
  llvmExtractor.def("NamesFromIR", &LE::NamesFromIR);
  llvmExtractor.def("InstsFromSource", &LE::InstsFromSource);
  llvmExtractor.def("InstsFromIR", &LE::InstsFromIR);
  llvmExtractor.def("WLCostFromSource", &LE::WLCostFromSource);
  llvmExtractor.def("WLCostFromIR", &LE::WLCostFromIR);
  llvmExtractor.def("IR2VecFromSource", &LE::IR2VecFromSource);
  llvmExtractor.def("IR2VecFromIR", &LE::IR2VecFromIR);

  // Subtypes
  py::module m = m_parent.def_submodule("llvm");

  // Graph extractor
  py::module m_graph = m.def_submodule("graph");

  py::class_<lg::ExtractionInfo, std::shared_ptr<lg::ExtractionInfo>>(
      m_graph, "ExtractionInfo")
      .def("accept", &lg::ExtractionInfo::accept)
      .def_readonly("functionInfos", &lg::ExtractionInfo::functionInfos)
      .def_readonly("callGraphInfo", &lg::ExtractionInfo::callGraphInfo);

  py::class_<lg::InstructionInfo, std::shared_ptr<lg::InstructionInfo>>(
      m_graph, "InstructionInfo")
      .def_readonly("instStr", &lg::InstructionInfo::instStr)
      .def_readonly("type", &lg::InstructionInfo::type)
      .def_readonly("opcode", &lg::InstructionInfo::opcode)
      .def_readonly("callTarget", &lg::InstructionInfo::callTarget)
      .def_readonly("isLoadOrStore", &lg::InstructionInfo::isLoadOrStore)
      .def_readonly("operands", &lg::InstructionInfo::operands)
      .def_readonly("function", &lg::InstructionInfo::function)
      .def_readonly("basicBlock", &lg::InstructionInfo::basicBlock)
      .def_readonly("recipThroughput", &lg::InstructionInfo::recipThroughput)
      .def_readonly("latency", &lg::InstructionInfo::latency)
      .def_readonly("codeSize", &lg::InstructionInfo::codeSize);

  py::class_<lg::MemoryAccessInfo, std::shared_ptr<lg::MemoryAccessInfo>>(
      m_graph, "MemoryAccessInfo")
      .def_readonly("type", &lg::MemoryAccessInfo::type)
      .def_readonly("inst", &lg::MemoryAccessInfo::inst)
      .def_readonly("basicBlock", &lg::MemoryAccessInfo::basicBlock)
      .def_readonly("dependencies", &lg::MemoryAccessInfo::dependencies);

  py::class_<lg::BasicBlockInfo, std::shared_ptr<lg::BasicBlockInfo>>(
      m_graph, "BasicBlockInfo")
      .def_readonly("name", &lg::BasicBlockInfo::name)
      .def_readonly("fullName", &lg::BasicBlockInfo::fullName)
      .def_readonly("instructions", &lg::BasicBlockInfo::instructions)
      .def_readonly("successors", &lg::BasicBlockInfo::successors)
      .def_readonly("frequency", &lg::BasicBlockInfo::frequency);

  py::class_<lg::FunctionInfo, std::shared_ptr<lg::FunctionInfo>>(
      m_graph, "FunctionInfo")
      .def("accept", &lg::FunctionInfo::accept)
      .def_readonly("name", &lg::FunctionInfo::name)
      .def_readonly("type", &lg::FunctionInfo::type)
      .def_readonly("entryInstruction", &lg::FunctionInfo::entryInstruction)
      .def_readonly("exitInstructions", &lg::FunctionInfo::exitInstructions)
      .def_readonly("args", &lg::FunctionInfo::args)
      .def_readonly("basicBlocks", &lg::FunctionInfo::basicBlocks)
      .def_readonly("memoryAccesses", &lg::FunctionInfo::memoryAccesses);

  py::class_<lg::CallGraphInfo, std::shared_ptr<lg::CallGraphInfo>>(
      m_graph, "CallGraphInfo")
      .def_readonly("calls", &lg::CallGraphInfo::calls);

  py::class_<lg::ArgInfo, std::shared_ptr<lg::ArgInfo>>(m_graph, "ArgInfo")
      .def_readonly("name", &lg::ArgInfo::name)
      .def_readonly("type", &lg::ArgInfo::type);

  py::class_<lg::ConstantInfo, std::shared_ptr<lg::ConstantInfo>>(m_graph, "ConstantInfo")
      .def_readonly("type", &lg::ConstantInfo::type)
      .def_readonly("value", &lg::ConstantInfo::value);

  // Sequence extractor
  py::module m_seq = m.def_submodule("seq");

  py::class_<ls::ExtractionInfo, std::shared_ptr<ls::ExtractionInfo>>(
      m_seq, "ExtractionInfo")
      .def("accept", &ls::ExtractionInfo::accept)
      .def_readonly("functionInfos", &ls::ExtractionInfo::functionInfos);

  py::class_<ls::FunctionInfo, std::shared_ptr<ls::FunctionInfo>>(
      m_seq, "FunctionInfo")
      .def("accept", &ls::FunctionInfo::accept)
      .def_readonly("name", &ls::FunctionInfo::name)
      .def_readonly("signature", &ls::FunctionInfo::signature)
      .def_readonly("basicBlocks", &ls::FunctionInfo::basicBlocks)
      .def_readonly("str", &ls::FunctionInfo::str);

  py::class_<ls::BasicBlockInfo, std::shared_ptr<ls::BasicBlockInfo>>(
      m_seq, "BasicBlockInfo")
      .def_readonly("name", &ls::BasicBlockInfo::name)
      .def_readonly("instructions", &ls::BasicBlockInfo::instructions);

  py::class_<ls::InstructionInfo, std::shared_ptr<ls::InstructionInfo>>(
      m_seq, "InstructionInfo")
      .def_readonly("tokens", &ls::InstructionInfo::tokens);

  // Names extractor
  py::module m_names = m.def_submodule("names");

  py::class_<ln::ExtractionInfo, std::shared_ptr<ln::ExtractionInfo>>(
      m_names, "ExtractionInfo")
      .def("accept", &ln::ExtractionInfo::accept)
      .def_readonly("globalInfos", &ln::ExtractionInfo::globalInfos)
      .def_readonly("functionInfos", &ln::ExtractionInfo::functionInfos);

  py::class_<ln::GlobalInfo, std::shared_ptr<ln::GlobalInfo>>(
      m_names, "GlobalInfo")
      .def("accept", &ln::GlobalInfo::accept)
      .def_readonly("name", &ln::GlobalInfo::name);

  py::class_<ln::FunctionInfo, std::shared_ptr<ln::FunctionInfo>>(
      m_names, "FunctionInfo")
      .def("accept", &ln::FunctionInfo::accept)
      .def_readonly("name", &ln::FunctionInfo::name)
      .def_readonly("signature", &ln::FunctionInfo::signature);

  // Milepost Static Features extractor
  py::module m_msf = m.def_submodule("msf");

  py::class_<lm::ExtractionInfo, std::shared_ptr<lm::ExtractionInfo>>(
      m_msf, "ExtractionInfo")
      .def("accept", &lm::ExtractionInfo::accept)
      .def_readonly("functionInfos", &lm::ExtractionInfo::functionInfos);

  py::class_<lm::FunctionInfo, std::shared_ptr<lm::FunctionInfo>>(
      m_msf, "FunctionInfo")
      .def("accept", &lm::FunctionInfo::accept)
      .def_readonly("name", &lm::FunctionInfo::name)
      .def_readonly("signature", &lm::FunctionInfo::signature)
      .def_readonly("features", &lm::FunctionInfo::features);

  // LLVM Instructions extractor
  py::module m_insts = m.def_submodule("insts");

  py::class_<li::ExtractionInfo, std::shared_ptr<li::ExtractionInfo>>(
      m_insts, "ExtractionInfo")
      .def("accept", &li::ExtractionInfo::accept)
      .def_readonly("functionInfos", &li::ExtractionInfo::functionInfos);

  py::class_<li::FunctionInfo, std::shared_ptr<li::FunctionInfo>>(
      m_insts, "FunctionInfo")
      .def("accept", &li::FunctionInfo::accept)
      .def_readonly("name", &li::FunctionInfo::name)
      .def_readonly("signature", &li::FunctionInfo::signature)
      .def_readonly("instructions", &li::FunctionInfo::instructions);

  // LLVM Wu-Larus Cost extractor
  py::module m_wlcost = m.def_submodule("wlcost");

  py::class_<lwlc::ExtractionInfo, std::shared_ptr<lwlc::ExtractionInfo>>(
      m_wlcost, "ExtractionInfo")
      .def("accept", &lwlc::ExtractionInfo::accept)
      .def_readonly("functionInfos", &lwlc::ExtractionInfo::functionInfos);

  py::class_<lwlc::FunctionInfo, std::shared_ptr<lwlc::FunctionInfo>>(
      m_wlcost, "FunctionInfo")
      .def("accept", &lwlc::FunctionInfo::accept)
      .def_readonly("name", &lwlc::FunctionInfo::name)
      .def_readonly("signature", &lwlc::FunctionInfo::signature)
      .def_readonly("recipThroughput", &lwlc::FunctionInfo::recipThroughput)
      .def_readonly("latency", &lwlc::FunctionInfo::latency)
      .def_readonly("codeSize", &lwlc::FunctionInfo::codeSize)
      .def_readonly("oneCost", &lwlc::FunctionInfo::oneCost);

  // IR2Vec extractor
  py::module m_ir2vec = m.def_submodule("ir2vec");

  py::class_<li2v::ExtractionInfo, std::shared_ptr<li2v::ExtractionInfo>>(
      m_ir2vec, "ExtractionInfo")
      .def("accept", &li2v::ExtractionInfo::accept)
      .def_readonly("functionInfos", &li2v::ExtractionInfo::functionInfos)
      .def_readonly("instructionInfos", &li2v::ExtractionInfo::instructionInfos)
      .def_readonly("moduleInfo", &li2v::ExtractionInfo::moduleInfo);

  py::class_<li2v::InstructionInfo, std::shared_ptr<li2v::InstructionInfo>>(
      m_ir2vec, "InstructionInfo")
      .def("accept", &li2v::InstructionInfo::accept)
      .def_readonly("instStr", &li2v::InstructionInfo::instStr)
      .def_readonly("opcode", &li2v::InstructionInfo::opcode)
      .def_readonly("ir2vec", &li2v::InstructionInfo::ir2vec);

  py::class_<li2v::FunctionInfo, std::shared_ptr<li2v::FunctionInfo>>(
      m_ir2vec, "FunctionInfo")
      .def("accept", &li2v::FunctionInfo::accept)
      .def_readonly("name", &li2v::FunctionInfo::name)
      .def_readonly("signature", &li2v::FunctionInfo::signature)
      .def_readonly("ir2vec", &li2v::FunctionInfo::ir2vec);

  py::class_<li2v::ModuleInfo, std::shared_ptr<li2v::ModuleInfo>>(
      m_ir2vec, "ModuleInfo")
      .def("accept", &li2v::ModuleInfo::accept)
      .def_readonly("name", &li2v::ModuleInfo::name)
      .def_readonly("ir2vec", &li2v::ModuleInfo::ir2vec);
}


PYBIND11_MODULE(extractors, m) {
  py::class_<IVisitor, PyVisitor>(m, "Visitor")
    .def(py::init<>());

  registerClangDriver(m);
  registerLLVMDriver(m);

  registerClangExtractor(m);
  registerLLVMExtractor(m);
}
