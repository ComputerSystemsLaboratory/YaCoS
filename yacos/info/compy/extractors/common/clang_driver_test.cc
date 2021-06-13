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

#include "clang_driver.h"

#include <iostream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;
using namespace compy;

class MockPass : public llvm::ModulePass {
 public:
  char ID = 0;
  MockPass() : llvm::ModulePass(ID) {}

  MOCK_METHOD1(runOnModule, bool(llvm::Module &M));
  MOCK_CONST_METHOD1(getAnalysisUsage, void(llvm::AnalysisUsage &au));
};

class ClangDriverFixture : public testing::Test {
 protected:
  void SetUp() override {
    // Init extractor
    std::vector<std::tuple<std::string, ClangDriver::IncludeDirType>>
        includeDirs = {
            std::make_tuple("/usr/include",
                            ClangDriver::IncludeDirType::SYSTEM),
            std::make_tuple("/usr/include/x86_64-linux-gnu",
                            ClangDriver::IncludeDirType::SYSTEM),
            std::make_tuple(
                "/devel/git_3rd/llvm-project/build_release/lib/clang/"
                "7.1.0/include/",
                ClangDriver::IncludeDirType::SYSTEM)};
    std::vector<std::string> compilerFlags = {"-Werror"};

    clang_.reset(new ClangDriver(ClangDriver::ProgrammingLanguage::C,
                                 ClangDriver::OptimizationLevel::O0,
                                 includeDirs, compilerFlags));
  }

  std::shared_ptr<ClangDriver> clang_;
};

// Tests
TEST_F(ClangDriverFixture, CompileWithPassFunction1) {
  NiceMock<MockPass> *pass = new NiceMock<MockPass>();
  EXPECT_CALL(*pass, runOnModule(_)).Times(AtLeast(1));

  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;
  passes.push_back(pass);

  clang_->Invoke("program_k1.c", frontendActions, passes);
}
