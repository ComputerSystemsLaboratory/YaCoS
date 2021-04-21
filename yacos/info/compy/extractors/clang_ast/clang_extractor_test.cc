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

#include <string>
#include <fstream>
#include <streambuf>

#include "clang_extractor.h"

#include "common/clang_driver.h"
#include "gtest/gtest.h"

using namespace testing;
using namespace compy;
using namespace compy::clang;

using CE = ClangExtractor;
using CD = ClangDriver;

class ClangExtractorFixture : public testing::Test {
 protected:
  void Init(CD::ProgrammingLanguage programmingLanguage) {
    // Init extractor
    std::vector<std::tuple<std::string, CD::IncludeDirType>> includeDirs = {
        std::make_tuple("/usr/include", CD::IncludeDirType::SYSTEM),
        std::make_tuple("/usr/include/x86_64-linux-gnu",
                        CD::IncludeDirType::SYSTEM),
        std::make_tuple("/usr/lib/llvm-10/lib/clang/10.0.0/include",
                        CD::IncludeDirType::SYSTEM),
        std::make_tuple("/usr/lib/llvm-10/lib/clang/10.0.1/include",
                        CD::IncludeDirType::SYSTEM)};
    std::vector<std::string> compilerFlags = {"-Werror"};

    driver_.reset(new ClangDriver(programmingLanguage,
                                  CD::OptimizationLevel::O0, includeDirs,
                                  compilerFlags));
    extractor_.reset(new CE(driver_));
  }

  std::shared_ptr<CD> driver_;
  std::shared_ptr<CE> extractor_;
};

class ClangExtractorCFixture : public ClangExtractorFixture {
 protected:
  void SetUp() override { Init(CD::ProgrammingLanguage::C); }
};

class ClangExtractorCPlusPlusFixture : public ClangExtractorFixture {
 protected:
  void SetUp() override { Init(CD::ProgrammingLanguage::CPLUSPLUS); }
};

//TEST_F(ClangExtractorCFixture, ExtractGraphFromFunction5) {
//  graph::ExtractionInfoPtr info = extractor_->GraphFromSource("program_k5.c");
//
//  ASSERT_EQ(info->functionInfos.size(), 2UL);
//}
//
//TEST_F(ClangExtractorCFixture, ExtractSeqFromFunction5) {
//  seq::ExtractionInfoPtr info = extractor_->SeqFromSource("program_k5.c");
//}
//
//TEST(O, WithOpenCL) {
//  std::shared_ptr<CD> driver_;
//  std::shared_ptr<CE> extractor_;
//
//  // Init extractor
//  std::vector<std::tuple<std::string, CD::IncludeDirType>> includeDirs = {
//      std::make_tuple("/usr/include", CD::IncludeDirType::SYSTEM),
//      std::make_tuple("/usr/include/x86_64-linux-gnu",
//                      CD::IncludeDirType::SYSTEM),
//      std::make_tuple("/devel/git_3rd/llvm-project/build_release/lib/clang/"
//                      "7.1.0/include/",
//                      CD::IncludeDirType::SYSTEM)};
//  std::vector<std::string> compilerFlags = {"-xcl"};
//
//  driver_.reset(new ClangDriver(CD::ProgrammingLanguage::OPENCL,
//                                CD::OptimizationLevel::O0, includeDirs,
//                                compilerFlags));
//  extractor_.reset(new CE(driver_));
//
//  std::ifstream t("/devel/git/research/code_graphs/eval/datasets/devmap/data/rodinia-3.1/opencl/leukocyte/OpenCL/track_ellipse_kernel_opt.cl");
//  std::string str((std::istreambuf_iterator<char>(t)),
//                  std::istreambuf_iterator<char>());
//  str = "#include \"/devel/git/gnns4code/c/3rd_party/opencl-shim.h\"\n" + str;
//
//  seq::ExtractionInfoPtr info = extractor_->SeqFromSource(str);
//}
