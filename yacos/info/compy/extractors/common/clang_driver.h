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
#include <string>
#include <tuple>
#include <vector>

#include "clang/Frontend/FrontendAction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace compy {

class ClangDriver {
 public:
  enum ProgrammingLanguage {
    C = 0,
    CPLUSPLUS = 1,
    OPENCL = 3
  };

  enum OptimizationLevel { O0 = 0, O1 = 1, O2 = 2, O3 = 3, Os = 4, Oz = 5};

  enum IncludeDirType {
    SYSTEM = 0,
    USER = 1,
  };

 public:
  ClangDriver(ProgrammingLanguage programmingLanguage,
              OptimizationLevel optimizationLevel,
              std::vector<std::tuple<std::string, IncludeDirType>> includeDirs,
              std::vector<std::string> compilerFlags);

  void addIncludeDir(std::string includeDir, IncludeDirType includeDirType);
  void removeIncludeDir(std::string includeDir, IncludeDirType includeDirType);
  void setOptimizationLevel(OptimizationLevel optimizationLevel);

  void Invoke(std::string filename, std::vector<::clang::FrontendAction *> frontendActions,
              std::vector<::llvm::Pass *> passes);

 private:
  std::shared_ptr<::llvm::legacy::PassManager> pm_;

  ProgrammingLanguage programmingLanguage_;
  OptimizationLevel optimizationLevel_;
  std::vector<std::tuple<std::string, IncludeDirType>> includeDirs_;
  std::vector<std::string> compilerFlags_;

};
using ClangDriverPtr = std::shared_ptr<ClangDriver>;

}  // namespace compy
