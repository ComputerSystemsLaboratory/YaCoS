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

#include "llvm_extractor.h"

#include <string>

#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/Compiler.h"
#include "llvm_graph_pass.h"
#include "llvm_seq_pass.h"
#include "llvm_msf_pass.h"
#include "llvm_loop_pass.h"
#include "llvm_names_pass.h"
#include "llvm_insts_pass.h"
#include "llvm_histogram_pass.h"
#include "llvm_opcodes_pass.h"
#include "llvm_wl_cost_pass.h"
#include "llvm_ir2vec_pass.h"

using namespace ::clang;
using namespace ::llvm;

namespace compy {
namespace llvm {

LLVMIRExtractor::LLVMIRExtractor(ClangDriverPtr clangDriver)
    : clangDriver_(clangDriver) {}

LLVMIRExtractor::LLVMIRExtractor(LLVMDriverPtr llvmDriver)
    : llvmDriver_(llvmDriver) {}

graph::ExtractionInfoPtr LLVMIRExtractor::GraphFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  //passes.push_back(createStripSymbolsPass());

  graph::ExtractorPass* extractorPass = new graph::ExtractorPass();
  passes.push_back(extractorPass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return extractorPass->extractionInfo;
}

graph::ExtractionInfoPtr LLVMIRExtractor::GraphFromIR(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  //passes.push_back(createStripSymbolsPass());

  graph::ExtractorPass* extractorPass = new graph::ExtractorPass();
  passes.push_back(extractorPass);

  llvmDriver_->Invoke(filename, passes);

  return extractorPass->extractionInfo;
}

seq::ExtractionInfoPtr LLVMIRExtractor::SeqFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  //passes.push_back(createStripSymbolsPass());

  seq::ExtractorPass *pass = new seq::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return pass->extractionInfo;
}

seq::ExtractionInfoPtr LLVMIRExtractor::SeqFromIR(std::string filename) {
  std::vector<::llvm::Pass *> passes;

  //passes.push_back(createStripSymbolsPass());

  seq::ExtractorPass *pass = new seq::ExtractorPass();
  passes.push_back(pass);

  llvmDriver_->Invoke(filename, passes);

  return pass->extractionInfo;
}

msf::ExtractionInfoPtr LLVMIRExtractor::MSFFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  msf::ExtractorPass *pass = new msf::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return pass->extractionInfo;
}

msf::ExtractionInfoPtr LLVMIRExtractor::MSFFromIR(std::string filename) {
  std::vector<::llvm::Pass *> passes;

  msf::ExtractorPass *pass = new msf::ExtractorPass();
  passes.push_back(pass);

  llvmDriver_->Invoke(filename, passes);

  return pass->extractionInfo;
}

loop::ExtractionInfoPtr LLVMIRExtractor::LoopFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  loop::ExtractorPass *pass = new loop::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return pass->extractionInfo;
}

loop::ExtractionInfoPtr LLVMIRExtractor::LoopFromIR(std::string filename) {
  std::vector<::llvm::Pass *> passes;

  loop::ExtractorPass *pass = new loop::ExtractorPass();
  passes.push_back(pass);

  llvmDriver_->Invoke(filename, passes);

  return pass->extractionInfo;
}

names::ExtractionInfoPtr LLVMIRExtractor::NamesFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  // Strip remove private constant. So do not use strip symbols.
  //passes.push_back(createStripSymbolsPass());
  names::ExtractorPass *pass = new names::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return pass->extractionInfo;
}

names::ExtractionInfoPtr LLVMIRExtractor::NamesFromIR(std::string filename) {
  std::vector<::llvm::Pass *> passes;

  names::ExtractorPass *pass = new names::ExtractorPass();
  passes.push_back(pass);

  llvmDriver_->Invoke(filename, passes);

  return pass->extractionInfo;
}

insts::ExtractionInfoPtr LLVMIRExtractor::InstsFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  insts::ExtractorPass *pass = new insts::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return pass->extractionInfo;
}

insts::ExtractionInfoPtr LLVMIRExtractor::InstsFromIR(std::string filename) {
  std::vector<::llvm::Pass *> passes;

  insts::ExtractorPass *pass = new insts::ExtractorPass();
  passes.push_back(pass);

  llvmDriver_->Invoke(filename, passes);

  return pass->extractionInfo;
}

histogram::ExtractionInfoPtr LLVMIRExtractor::HistogramFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  histogram::ExtractorPass *pass = new histogram::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return pass->extractionInfo;
}

histogram::ExtractionInfoPtr LLVMIRExtractor::HistogramFromIR(std::string filename) {
  std::vector<::llvm::Pass *> passes;

  histogram::ExtractorPass *pass = new histogram::ExtractorPass();
  passes.push_back(pass);

  llvmDriver_->Invoke(filename, passes);

  return pass->extractionInfo;
}

opcodes::ExtractionInfoPtr LLVMIRExtractor::OpcodesFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  opcodes::ExtractorPass *pass = new opcodes::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return pass->extractionInfo;
}

opcodes::ExtractionInfoPtr LLVMIRExtractor::OpcodesFromIR(std::string filename) {
  std::vector<::llvm::Pass *> passes;

  opcodes::ExtractorPass *pass = new opcodes::ExtractorPass();
  passes.push_back(pass);

  llvmDriver_->Invoke(filename, passes);

  return pass->extractionInfo;
}

wlcost::ExtractionInfoPtr LLVMIRExtractor::WLCostFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  wlcost::ExtractorPass *pass = new wlcost::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return pass->extractionInfo;
}

wlcost::ExtractionInfoPtr LLVMIRExtractor::WLCostFromIR(std::string filename) {
  std::vector<::llvm::Pass *> passes;

  wlcost::ExtractorPass *pass = new wlcost::ExtractorPass();
  passes.push_back(pass);

  llvmDriver_->Invoke(filename, passes);

  return pass->extractionInfo;
}

ir2vec::ExtractionInfoPtr LLVMIRExtractor::IR2VecFromSource(std::string filename) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  ir2vec::ExtractorPass *pass = new ir2vec::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return pass->extractionInfo;
}

ir2vec::ExtractionInfoPtr LLVMIRExtractor::IR2VecFromIR(std::string filename) {
  std::vector<::llvm::Pass *> passes;

  ir2vec::ExtractorPass *pass = new ir2vec::ExtractorPass();
  passes.push_back(pass);

  llvmDriver_->Invoke(filename, passes);

  return pass->extractionInfo;
}

}  // namespace llvm
}  // namespace compy
