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
#include <sstream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"

using namespace ::clang;
using namespace ::llvm;

namespace compy {

static void LLVMErrorHandler(void *UserData, const std::string &Message,
                             bool GenCrashDiag) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine *>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // Run the interrupt handlers to make sure any special cleanups get done, in
  // particular that we remove files registered with RemoveFileOnSignal.
  ::llvm::sys::RunInterruptHandlers();

  // We cannot recover from llvm errors.  When reporting a fatal error, exit
  // with status 70 to generate crash diagnostics.  For BSD systems this is
  // defined as an internal software error.  Otherwise, exit with status 1.
  exit(GenCrashDiag ? 70 : 1);
}

ClangDriver::ClangDriver(
    ProgrammingLanguage programmingLanguage,
    OptimizationLevel optimizationLevel,
    std::vector<std::tuple<std::string, IncludeDirType>> includeDirs,
    std::vector<std::string> compilerFlags)
    : programmingLanguage_(programmingLanguage),
      optimizationLevel_(optimizationLevel),
      includeDirs_(includeDirs),
      compilerFlags_(compilerFlags) {}

void ClangDriver::addIncludeDir(std::string includeDir,
                                IncludeDirType includeDirType) {
  includeDirs_.insert(includeDirs_.begin(), std::make_tuple(includeDir, includeDirType));
}

void ClangDriver::removeIncludeDir(std::string includeDir,
                                   IncludeDirType includeDirType) {
  includeDirs_.erase(std::remove(includeDirs_.begin(), includeDirs_.end(), std::make_tuple(includeDir, includeDirType)),
  includeDirs_.end());
}

void ClangDriver::setOptimizationLevel(OptimizationLevel optimizationLevel) {
  optimizationLevel_ = optimizationLevel;
}

void ClangDriver::Invoke(std::string filename, std::vector<::clang::FrontendAction *> frontendActions,
                         std::vector<::llvm::Pass *> passes) {
    const char *name;
    switch (programmingLanguage_) {
        case ProgrammingLanguage::C:
            name = "program.c";
            break;
        case ProgrammingLanguage::CPLUSPLUS:
            name = "program.cc";
            break;
        case ProgrammingLanguage::OPENCL:
            name = "program.cl";
            break;
    }

    std::ifstream t(filename);
    std::stringstream source_code;
    source_code << t.rdbuf();
    std::string str = source_code.str();
    auto code = str.c_str();

    std::vector<const char *> args;
    args.push_back(name);

    // Optimization level.
    const char *optimizationLevelChr;
    switch (optimizationLevel_) {
        case OptimizationLevel::O0:
            optimizationLevelChr = "-O0";
            break;
        case OptimizationLevel::O1:
            optimizationLevelChr = "-O1";
            break;
        case OptimizationLevel::O2:
            optimizationLevelChr = "-O2";
            break;
        case OptimizationLevel::O3:
            optimizationLevelChr = "-O3";
            break;
        case OptimizationLevel::Os:
            optimizationLevelChr = "-Os";
            break;
        case OptimizationLevel::Oz:
            optimizationLevelChr = "-Oz";
            break;
    }
    args.push_back(optimizationLevelChr);

    // Additional flags.
    for (auto flag : compilerFlags_) {
        args.push_back(flag.c_str());
    }

    std::unique_ptr<clang::CompilerInstance> Clang(new CompilerInstance());
    IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

    // Register the support for object-file-wrapped LLVMIRExtractor modules.
    auto PCHOps = Clang->getPCHContainerOperations();
    PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
    PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

    // Initialize targets first, so that --version shows registered targets.
    ::llvm::InitializeAllTargets();
    ::llvm::InitializeAllTargetMCs();
    ::llvm::InitializeAllAsmPrinters();
    ::llvm::InitializeAllAsmParsers();

    // Buffer diagnostics from argument parsing so that we can output them using a
    // well formed diagnostic object.
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
    TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

    // Initialize CompilerInvocation.
    CompilerInvocation::CreateFromArgs(Clang->getInvocation(), ArrayRef<const char *>(args), Diags);

    // Map code name to a memoryBuffer.
    StringRef codeData(code);
    std::unique_ptr<MemoryBuffer> buffer =
            MemoryBuffer::getMemBufferCopy(codeData);
    Clang->getInvocation().getPreprocessorOpts().addRemappedFile(
            name, buffer.release());

    // Add include paths.
    for (auto includeDir : includeDirs_) {
        frontend::IncludeDirGroup includeDirGroup;
        switch (std::get<1>(includeDir)) {
            case IncludeDirType::SYSTEM:
                includeDirGroup = frontend::IncludeDirGroup::System;
                break;
            case IncludeDirType::USER:
                includeDirGroup = frontend::IncludeDirGroup::Angled;
                break;
        }

        Clang->getHeaderSearchOpts().UserEntries.push_back(
                HeaderSearchOptions::Entry(std::get<0>(includeDir), includeDirGroup,
                                           false, false));
    }

    // Create the actual diagnostics engine.
    Clang->createDiagnostics();

    // Set an error handler, so that any LLVM backend diagnostics go through our
    // error handler.
    ::llvm::install_fatal_error_handler(
            LLVMErrorHandler, static_cast<void *>(&Clang->getDiagnostics()));

    DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());

    // Run clang frontend actions.
    for (auto frontendAction : frontendActions) {
        if (!Clang->ExecuteAction(*frontendAction)) {
            for (TextDiagnosticBuffer::const_iterator I = DiagsBuffer->err_begin(),
                         E = DiagsBuffer->err_end();
                 I != E; ++I)
                std::cout << "# " << I->second << '\n';

            throw std::runtime_error("Failed compiling to execute frontend action");
        }
    }

    // Convert to LLVM module if needed (if there are any LLVM passes).
    if (!passes.empty()) {
        // Lower Clang AST to LLVM bitcode module.
        std::unique_ptr<CodeGenAction> Act(new EmitLLVMOnlyAction());
        if (!Clang->ExecuteAction(*Act)) {
            for (TextDiagnosticBuffer::const_iterator I = DiagsBuffer->err_begin(),
                         E = DiagsBuffer->err_end();
                 I != E; ++I)
                std::cout << "# " << I->second << '\n';

            throw std::runtime_error("Failed compiling to LLVM module");
        }
        std::unique_ptr<::llvm::Module> Module = Act->takeModule();

        ::llvm::remove_fatal_error_handler();

        // Register other llvm passes.
        PassRegistry &reg = *PassRegistry::getPassRegistry();
        initializeCallGraphWrapperPassPass(reg);
        initializeMemorySSAWrapperPassPass(reg);
        initializeStripSymbolsPass(reg);

        // Setup the pass manager and add passes.
        pm_.reset(new legacy::PassManager());
        for (auto pass : passes) {
            pm_->add(pass);
        }

        // Run passes.
        pm_->run(*Module);
    }
}

}  // namespace compy
