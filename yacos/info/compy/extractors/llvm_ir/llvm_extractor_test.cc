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

#include <fstream>
#include <iostream>

#include "gtest/gtest.h"

using namespace ::llvm;
using namespace compy;
using namespace compy::llvm;

using LE = LLVMIRExtractor;
using CD = ClangDriver;

constexpr char kProgram4ForwardDecl[] = "int barbara(float x, float y);";

void createFileWithContents(std::string filename, std::string filecontent) {
  std::ofstream tempHeaderFile(filename.c_str());
  tempHeaderFile << filecontent << std::endl;
  tempHeaderFile.close();
}
void removeFile(std::string filename) { std::remove(filename.c_str()); }

class LLVMExtractorFixture : public testing::Test {
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
    extractor_.reset(new LE(driver_));
  }

  std::shared_ptr<CD> driver_;
  std::shared_ptr<LE> extractor_;
};

class LLVMExtractorCFixture : public LLVMExtractorFixture {
 protected:
  void SetUp() override { Init(CD::ProgrammingLanguage::C); }
};

class LLVMExtractorCPlusPlusFixture : public LLVMExtractorFixture {
 protected:
  void SetUp() override { Init(CD::ProgrammingLanguage::CPLUSPLUS); }
};

// C tests

TEST_F(LLVMExtractorCFixture, ExtractFromFunction1) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k1.c";

  graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "foo");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromFunction2) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k2.c";
  graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "max");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 2UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromFunction5) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k5.c";
  graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);

  ASSERT_EQ(info->functionInfos.size(), 2UL);
  ASSERT_EQ(info->functionInfos[0]->name, "max");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 2UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromFunctionWithSystemInclude) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k3.c";
  graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "foo");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromFunctionWithUserInclude) {
  std::string headerFilename = "/tmp/tempHdr.h";
  createFileWithContents(headerFilename, kProgram4ForwardDecl);

  driver_->addIncludeDir("/tmp", CD::IncludeDirType::SYSTEM);

  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k4.c";

  graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);

  removeFile(headerFilename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromNoFunction) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromSource("");

  ASSERT_EQ(info->functionInfos.size(), 0UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromBadFunction) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_foo.c";

  EXPECT_THROW(
      {
        try {
          graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);
        } catch (std::runtime_error const& err) {
          EXPECT_EQ(err.what(), std::string("Failed compiling to LLVM module"));
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(LLVMExtractorCFixture, ExtractWithDifferentOptimizationlevels) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k2.c";

  driver_->setOptimizationLevel(CD::OptimizationLevel::O0);
  graph::ExtractionInfoPtr infoO0 = extractor_->GraphFromSource(filename);

  driver_->setOptimizationLevel(CD::OptimizationLevel::O1);
  graph::ExtractionInfoPtr infoO1 = extractor_->GraphFromSource(filename);

  ASSERT_EQ(infoO0->functionInfos.size(), 1UL);
  ASSERT_EQ(infoO1->functionInfos.size(), 1UL);
  ASSERT_TRUE(infoO0->functionInfos[0]->basicBlocks.size() >
              infoO1->functionInfos[0]->basicBlocks.size());
}

// C++ tests
TEST_F(LLVMExtractorCPlusPlusFixture, ExtractFromFunction1) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k1.c";
  graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "_Z3foov");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCPlusPlusFixture, ExtractFromFunction2) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k2.c";
  graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "_Z3maxii");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 2UL);
}

TEST_F(LLVMExtractorCPlusPlusFixture, ExtractFromFunctionWithSystemInclude) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k3.c";
  graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "_Z3foov");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCPlusPlusFixture, ExtractFromFunctionWithUserInclude) {
  const char* env_p = std::getenv("HOME");
  std::string filename;
  filename = env_p;
  filename = filename + "/.local/yacos/tests/program_k4.c";

  std::string headerFilename = "/tmp/tempHdr.h";
  createFileWithContents(headerFilename, kProgram4ForwardDecl);

  driver_->addIncludeDir("/tmp", CD::IncludeDirType::SYSTEM);
  graph::ExtractionInfoPtr info = extractor_->GraphFromSource(filename);

  removeFile(headerFilename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}
