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

#include "gtest/gtest.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm_graph_pass.h"
#include "llvm_seq_pass.h"

using namespace llvm;
using namespace compy;
using namespace compy::llvm;

using std::string;

// LLVM samples
constexpr char kLLVM1[] =
    "define dso_local void @A(i32*) #0 {\n"
    "  %2 = alloca i32*, align 8\n"
    "  %3 = alloca i32, align 4\n"
    "  store i32* %0, i32** %2, align 8\n"
    "  store i32 2, i32* %3, align 4\n"
    "  %4 = load i32, i32* %3, align 4\n"
    "  %5 = load i32*, i32** %2, align 8\n"
    "  %6 = getelementptr inbounds i32, i32* %5, i64 0\n"
    "  store i32 %4, i32* %6, align 4\n"
    "  ret void\n"
    "}\n";
constexpr char kLLVM2[] =
    "define dso_local void @A(i32*) #0 {\n"
    "  %2 = alloca i32*, align 8\n"
    "  %3 = alloca i32, align 4\n"
    "  store i32* %0, i32** %2, align 8\n"
    "}\n";

class LLVMGraphPassFixture : public testing::Test {
 protected:
  void SetUp() override {
    // Register other llvm passes
    PassRegistry& reg = *PassRegistry::getPassRegistry();
    initializeCallGraphWrapperPassPass(reg);
    initializeMemorySSAWrapperPassPass(reg);

    // Setup the pass manager, add pass
    _pm = new legacy::PassManager();
    _ep = new graph::ExtractorPass();
    _pm->add(_ep);
  }

  void TearDown() override {
    free(_pm);
    free(_ep);
  }

  graph::ExtractionInfoPtr Extract(std::string ir) {
    // Construct an IR file from the filename passed on the command line.
    SMDiagnostic err;
    LLVMContext context;
    MemoryBufferRef mb = MemoryBuffer::getMemBuffer(ir)->getMemBufferRef();
    std::unique_ptr<Module> module = parseIR(mb, err, context);
    if (!module.get()) {
      throw std::runtime_error("Failed compiling to LLVM module");
    }

    // Run pass
    _pm->run(*module);

    // Return extraction info
    return _ep->extractionInfo;
  }

  legacy::PassManager* _pm;
  graph::ExtractorPass* _ep;
};

class LLVMSeqPassFixture : public testing::Test {
 protected:
  void SetUp() override {
    // Register other llvm passes
    PassRegistry& reg = *PassRegistry::getPassRegistry();
    initializeCallGraphWrapperPassPass(reg);
    initializeMemorySSAWrapperPassPass(reg);

    // Setup the pass manager, add pass
    _pm = new legacy::PassManager();
    _ep = new seq::ExtractorPass();
    _pm->add(_ep);
  }

  void TearDown() override {
    free(_pm);
    free(_ep);
  }

  seq::ExtractionInfoPtr Extract(std::string ir) {
    // Construct an IR file from the filename passed on the command line.
    SMDiagnostic err;
    LLVMContext context;
    MemoryBufferRef mb = MemoryBuffer::getMemBuffer(ir)->getMemBufferRef();
    std::unique_ptr<Module> module = parseIR(mb, err, context);
    if (!module.get()) {
      throw std::runtime_error("Failed compiling to LLVM module");
    }

    // Run pass
    _pm->run(*module);

    // Return extraction info
    return _ep->extractionInfo;
  }

  legacy::PassManager* _pm;
  seq::ExtractorPass* _ep;
};

TEST_F(LLVMGraphPassFixture, RunPassAndRetrieveSuccess) {
  graph::ExtractionInfoPtr info = Extract(kLLVM1);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
}

TEST_F(LLVMSeqPassFixture, RunPassAndRetrieveSuccess2) {
  seq::ExtractionInfoPtr info = Extract(kLLVM1);

  ASSERT_EQ(info->functionInfos.size(), 1UL);

  std::vector<std::string> signature = info->functionInfos[0]->signature;

  seq::BasicBlockInfoPtr basicBlock = info->functionInfos[0]->basicBlocks[0];
  ASSERT_GT(basicBlock->instructions.size(), 1UL);

  seq::InstructionInfoPtr instructionInfoPtr = basicBlock->instructions[0];
  ASSERT_GT(instructionInfoPtr->tokens.size(), 1UL);
}

TEST_F(LLVMGraphPassFixture, RunPassAndRetrieveFail) {
  EXPECT_THROW(
      {
        try {
          graph::ExtractionInfoPtr info = Extract(kLLVM2);
        } catch (std::runtime_error const& err) {
          EXPECT_EQ(err.what(), std::string("Failed compiling to LLVM module"));
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(LLVMGraphPassFixture, RunPassAndRetrieveZero) {
  graph::ExtractionInfoPtr info = Extract("");

  ASSERT_EQ(info->functionInfos.size(), 0UL);
}
