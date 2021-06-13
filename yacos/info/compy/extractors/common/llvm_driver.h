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

class LLVMDriver {
 public:
  LLVMDriver(std::vector<std::string> optimizations);
  LLVMDriver();
  void addOptimizationFront(std::string optimization);
  void addOptimizationsFront(std::vector<std::string> optimizations);
  void addOptimizationBack(std::string optimization);
  void addOptimizationsBack(std::vector<std::string> optimizations);
  void removeOptimization(std::string optimization);
  void removeOptimizationsFront(int optimizations);
  void removeOptimizationsBack(int optimizations);
  void setOptimizations(std::vector<std::string> optimizations);
  std::vector<std::string> getOptimizations();
  void clearOptimizations();
  void Invoke(std::string filename, std::vector<::llvm::Pass *> passes);

 private:
  void addOptimizationPasses();
  std::shared_ptr<::llvm::legacy::PassManager> pm_;
  std::vector<std::string> optimizations_;
};

using LLVMDriverPtr = std::shared_ptr<LLVMDriver>;

}  // namespace compy
