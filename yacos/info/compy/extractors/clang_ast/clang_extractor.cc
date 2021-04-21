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

#include "clang_extractor.h"

#include <string>

#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/Compiler.h"

#include "clang_seq_frontendaction.h"
#include "clang_graph_frontendaction.h"

using namespace ::clang;
using namespace ::llvm;

namespace compy {
namespace clang {

ClangExtractor::ClangExtractor(ClangDriverPtr clangDriver)
    : clangDriver_(clangDriver) {}

graph::ExtractionInfoPtr ClangExtractor::GraphFromSource(std::string filename) {
  compy::clang::graph::ExtractorFrontendAction* fa =
      new compy::clang::graph::ExtractorFrontendAction();

  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  frontendActions.push_back(fa);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return fa->extractionInfo;
}

seq::ExtractionInfoPtr ClangExtractor::SeqFromSource(std::string filename) {
  compy::clang::seq::ExtractorFrontendAction* fa =
      new compy::clang::seq::ExtractorFrontendAction();

  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  frontendActions.push_back(fa);

  clangDriver_->Invoke(filename, frontendActions, passes);

  return fa->extractionInfo;
}

}  // namespace clang
}  // namespace compy
