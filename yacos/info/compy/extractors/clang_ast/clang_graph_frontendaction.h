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
#include <vector>

#include <clang/Analysis/CFG.h>
#include "clang/AST/AST.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang_extractor.h"
#include "llvm/ADT/StringRef.h"

namespace compy {
namespace clang {
namespace graph {

class ExtractorASTVisitor
    : public ::clang::RecursiveASTVisitor<ExtractorASTVisitor> {
 public:
  ExtractorASTVisitor(::clang::ASTContext &context,
                      ExtractionInfoPtr extractionInfo)
      : context_(context), extractionInfo_(extractionInfo) {}

  bool VisitStmt(::clang::Stmt *s);
  bool VisitFunctionDecl(::clang::FunctionDecl *f);

 private:
  FunctionInfoPtr getInfo(const ::clang::FunctionDecl &func);
  CFGBlockInfoPtr getInfo(const ::clang::CFGBlock &block);
  StmtInfoPtr getInfo(const ::clang::Stmt &stmt);
  DeclInfoPtr getInfo(const ::clang::Decl &decl);

 private:
  ::clang::ASTContext &context_;
  ExtractionInfoPtr extractionInfo_;

  std::unordered_map<const ::clang::Stmt *, StmtInfoPtr> stmtInfos_;
  std::unordered_map<const ::clang::CFGBlock *, CFGBlockInfoPtr> cfgBlockInfos_;
  std::unordered_map<const ::clang::Decl *, DeclInfoPtr> declInfos_;
};

class ExtractorASTConsumer : public ::clang::ASTConsumer {
 public:
  ExtractorASTConsumer(::clang::ASTContext &context,
                       ExtractionInfoPtr extractionInfo);

  bool HandleTopLevelDecl(::clang::DeclGroupRef DR) override;

 private:
  ExtractorASTVisitor visitor_;
};

class ExtractorFrontendAction : public ::clang::ASTFrontendAction {
 public:
  std::unique_ptr<::clang::ASTConsumer> CreateASTConsumer(
      ::clang::CompilerInstance &CI, ::llvm::StringRef file) override;

  ExtractionInfoPtr extractionInfo;
};

}  // namespace graph
}  // namespace clang
}  // namespace compy
