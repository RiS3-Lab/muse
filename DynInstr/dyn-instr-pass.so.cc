/*
  DynInstr instrumentation pass
  ---------------------------------------------------

  Based on llvm-mode pass written by Laszlo Szekeres <lszekeres@google.com> and
  Michal Zalewski <lcamtuf@google.com>

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at:

  http://www.apache.org/licenses/LICENSE-2.0

  This library is plugged into LLVM when invoking clang through dyn-instr-clang.

*/

#define LLVM_PASS

#include "config.h"
#include "debug.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <fstream>

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Instruction.h"

using namespace llvm;

namespace {

  class DynInstr : public ModulePass {

  public:

    static char ID;
    DynInstr() : ModulePass(ID) { }
    bool seeded = false;
    bool runOnModule(Module &M) override;

    const char* getPassName() const override {
     return "Muse DynInstr Instrumentation";
    }

  };

}


char DynInstr::ID = 0;
static inline bool starts_with(const std::string& str, const std::string& prefix)
{
  if(prefix.length() > str.length()) { return false; }
  return str.substr(0, prefix.length()) == prefix;
}

static inline bool is_sanitizer_handler(BasicBlock& bb)
{
    // ubsan
    if (starts_with(bb.getName(), "handler.")) return true;
    // asan
    Instruction* inst = bb.getFirstNonPHI();
    if ((isa<CallInst>(inst))) {
        auto call = cast<CallInst>(inst);
        return starts_with(call->getCalledFunction()->getName(), "__asan_");
    }
    // everything else
    return false;

}

#include <iostream>
static inline bool is_llvm_dbg_intrinsic(Instruction& instr)
{
  const bool is_call = instr.getOpcode() == Instruction::Invoke ||
    instr.getOpcode() == Instruction::Call;
  if(!is_call) { return false; }

  CallSite cs(&instr);
  Function* calledFunc = cs.getCalledFunction();

  if (calledFunc != NULL) {
    const bool ret = calledFunc->isIntrinsic() &&
      starts_with(calledFunc->getName().str(), "llvm.");
    return ret;
  } else {
    return false;
  }
}

static inline int
count_sanitizer_instrumentation(BasicBlock &bb){
  int numSanitizerInst = 0;
  for (Instruction& inst : bb.getInstList()) {
    if (inst.getMetadata("afl_edge_sanitizer") != NULL) {
      numSanitizerInst++;
    }
  }
  return numSanitizerInst;
}

/* Returns how many lava label were instrumented.*/
// static int
// insert_lava_label(BasicBlock &bb, LLVMContext &C)
// {
//   int lava_calls = 0;
//   for (auto &ins: bb){
//     if (is_llvm_dbg_intrinsic(ins)) continue;
//     CallInst* call_inst = dyn_cast<CallInst>(&ins);
//     if(call_inst) {
//       Function* callee = dyn_cast<Function>(call_inst->getCalledValue());
//       if (callee) {
//         // OKF("function name %s", callee->getName().str().c_str());
//         if (!callee->getName().str().compare("lava_get")) {
//           auto meta_lava = MDNode::get(C, llvm::MDString::get(C, "lava"));
//           ins.setMetadata("afl_edge_sanitizer", meta_lava);
//           lava_calls++;
//         }
//       }
//     }
//   }
//   return lava_calls;
// }

static inline unsigned int
get_block_id(BasicBlock &bb)
{
  unsigned int bbid = 0;
  MDNode *bb_node = nullptr;
  for (auto &ins: bb){
    if ((bb_node = ins.getMetadata("afl_cur_loc"))) break;
  }
  if (bb_node){
    bbid = cast<ConstantInt>(cast<ValueAsMetadata>(bb_node->getOperand(0))->getValue())->getZExtValue();
  }
  return bbid;
}

// static inline unsigned int
// get_edge_id(BasicBlock &src, BasicBlock &dst)
// {
//   unsigned int src_bbid = 0, dst_bbid = 0;
//   src_bbid = get_block_id(src);
//   dst_bbid = get_block_id(dst);
//   if (src_bbid && dst_bbid){
//     return ((src_bbid >> 1) ^ dst_bbid);
//   }
//   return 0;
// }

bool isIndirectCall(CallInst *ci) {
    const Value *V = ci->getCalledValue();
    if (isa<Function>(V) || isa<Constant>(V))
        return false;
    if (const CallInst *CI = dyn_cast<CallInst>(ci))
        if (CI->isInlineAsm())
            return false;
    return true;
}


bool DynInstr::runOnModule(Module &M) {
  std::ofstream locMap("locmap.csv", std::ofstream::out);
  std::ofstream labelMap("labelmap.csv", std::ofstream::out);

  if(!seeded) {
    char* random_seed_str = getenv("AFL_RANDOM_SEED");
    if(random_seed_str != NULL) {
      unsigned int seed;
      sscanf(random_seed_str, "%u", &seed);
      srandom(seed);
      SAYF("seeded with %u\n", seed);
      seeded = true;
    }
  }
  const bool annotate_for_se = (getenv("ANNOTATE_FOR_SE") != NULL);
  LLVMContext &C = M.getContext();

  // IntegerType *Int8Ty  = IntegerType::getInt8Ty(C);
  IntegerType *Int32Ty = IntegerType::getInt32Ty(C);
  Type *VoidType = Type::getVoidTy(C);
  // PointerType *Ptr8Ty = Type::getInt8PtrTy(C, 0);
  Type* ArgInt32TyInt32Ty[] = {Int32Ty, Int32Ty};
  Type* ArgInt32Ty[] = {Int32Ty};


  /* Decide instrumentation ratio */

  char* inst_ratio_str = getenv("AFL_INST_RATIO");
  unsigned int inst_ratio = 100;

  if (inst_ratio_str) {

    if (sscanf(inst_ratio_str, "%u", &inst_ratio) != 1 || !inst_ratio ||
        inst_ratio > 100)
      FATAL("Bad value of AFL_INST_RATIO (must be between 1 and 100)");

  }

  /* Get globals for the previous location. Note that
     __afl_prev_loc is thread-local. */
  // GlobalVariable *AFLMapPtr;
  GlobalVariable *AFLPrevLoc;
  Function *AFLLogLoc;
  Function *AFLLogLabel;
  Function *AFLLogTriggered;
  Function *AFLLogCmp;
  Function *AFLLogExtCall;
  Function *AFLLogIndCall;
  Function *AFLLogMemOp;

  if (!annotate_for_se) {
    AFLPrevLoc = new GlobalVariable(
                                    M, Int32Ty, false, GlobalValue::ExternalLinkage, 0, "__afl_prev_loc",
                                    0, GlobalVariable::GeneralDynamicTLSModel, 0, false);

    AFLLogLoc = Function::Create(FunctionType::get(VoidType, ArrayRef<Type*>(ArgInt32TyInt32Ty), false), GlobalValue::ExternalLinkage, "__afl_log_loc", &M);
    AFLLogLabel = Function::Create(FunctionType::get(VoidType, ArrayRef<Type*>(ArgInt32Ty), false), GlobalValue::ExternalLinkage, "__afl_log_label", &M);
    AFLLogTriggered = Function::Create(FunctionType::get(VoidType, ArrayRef<Type*>(ArgInt32Ty),false), GlobalValue::ExternalLinkage, "__afl_log_triggered", &M);
    AFLLogCmp = Function::Create(FunctionType::get(VoidType, ArrayRef<Type*>(ArgInt32Ty), false),
                                 GlobalValue::ExternalLinkage, "__afl_log_cmp", &M);
    AFLLogExtCall = Function::Create(FunctionType::get(VoidType, ArrayRef<Type*>(ArgInt32Ty), false),
                                     GlobalValue::ExternalLinkage, "__afl_log_ext_call", &M);
    AFLLogIndCall = Function::Create(FunctionType::get(VoidType, ArrayRef<Type*>(ArgInt32Ty), false),
                                     GlobalValue::ExternalLinkage, "__afl_log_ind_call", &M);
    AFLLogMemOp = Function::Create(FunctionType::get(VoidType, ArrayRef<Type*>(ArgInt32Ty), false),
                                     GlobalValue::ExternalLinkage, "__afl_log_memop", &M);
  }

  /* Insert LAVA labels if required.*/
  if (getenv("INSERT_LAVA_LABEL")) {
    // int total_lava_num = 0;
#if 0
    // both binary and bc mode instrumentation should contain the metadata.
    // Align with UBSAN labels.
    for (auto &F : M) {
      for (auto &BB : F) {
        total_lava_num += insert_lava_label(BB, C);
      }
    }
    OKF("Instrumented %d lava labels", total_lava_num);
#else
    llvm::Function* lava_get = M.getFunction("lava_get");
    if (lava_get) {
      for (auto& BB : *lava_get) {
        auto meta_lava = MDNode::get(C, llvm::MDString::get(C, "lava"));
        (*BB.getTerminator()).setMetadata("afl_edge_sanitizer", meta_lava);
      }
    }
#endif
  }

  /* Instrument all the things! */

  int inst_blocks = 0;

  for (auto &F : M)
    for (auto &BB : F) {

      BasicBlock::iterator IP = BB.getFirstInsertionPt();
      IRBuilder<> IRB(&(*IP));

      if (AFL_R(100) >= inst_ratio) continue;

      /* Make up cur_loc */

      unsigned int cur_loc = AFL_R(MAP_SIZE);

      ConstantInt *CurLoc = ConstantInt::get(Int32Ty, cur_loc);

      bool has_savior_label = (count_sanitizer_instrumentation(BB) > 0);

      auto meta_loc = MDNode::get(C, ConstantAsMetadata::get(CurLoc));
      for (Instruction& instr : BB.getInstList()) {
        if (!is_llvm_dbg_intrinsic(instr)) {
          // this only insert the meta for the first non-llvm dbg
          instr.setMetadata("afl_cur_loc", meta_loc);
          break;
        }
      }
      if (annotate_for_se) {
        bool locMapped = false;
        for (Instruction& instr : BB.getInstList()) {
          if (MDNode* dbg = instr.getMetadata("dbg")) {
            DebugLoc Loc = DebugLoc(dbg);
            locMap << cur_loc << ","
                   << Loc->getDirectory().str() << ","
                   << Loc->getFilename().str() << ","
                   << Loc.getLine() << "\n";
            if (has_savior_label) {
              labelMap << cur_loc << ","
                       << Loc->getFilename().str() << ":"
                       << Loc.getLine() << "\n";
            }
            locMapped = true;
            break;
          }
        }
        if (!locMapped) {
          locMap << cur_loc << ",?,?,0\n";
        }
      } else  {
        /* Load prev_loc */
        LoadInst *PrevLoc = IRB.CreateLoad(AFLPrevLoc);
        PrevLoc->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
        Value *PrevLocCasted = IRB.CreateZExt(PrevLoc, IRB.getInt32Ty());

        /* XOR the cur_loc with prev_loc */
        Value* Xor = IRB.CreateXor(PrevLocCasted, CurLoc);


        /* log bbid and eid*/
        Value* ArgLoc[] = { CurLoc, Xor};
        IRB.CreateCall(AFLLogLoc, ArrayRef<Value*>(ArgLoc));

        // The updating the prev_loc will be the same either for SE or for AFL
        /* Set prev_loc to cur_loc >> 1 */
        StoreInst *Store =
          IRB.CreateStore(ConstantInt::get(Int32Ty, cur_loc >> 1), AFLPrevLoc);
        Store->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      }

      inst_blocks++;

    }

  /*log cmp and ext/ind calls*/
  if (!annotate_for_se) {
      int total_cmp = 0;
      int total_san = 0;
      int total_indcall = 0;
      int total_extcall = 0;
      int total_memop = 0;

      for (auto &F : M) {
          for (auto &BB : F) {
              BasicBlock::iterator IP = BB.getFirstInsertionPt();
              IRBuilder<> IRB(&(*IP));
              int bb_cmp = 0;
              int bb_indcall = 0;
              int bb_extcall = 0;
              int bb_san = 0;
              int bb_memop = 0;
              for (auto &I : BB) {
                  if (dyn_cast<CmpInst>(&I)) {
                      // check if it is a cmp
                      bb_cmp++;
                  }

                  if (auto call = dyn_cast<CallInst>(&I)) {
                      if (isIndirectCall(call)) {
                          // check if it is indirect call
                          bb_indcall++;
                      } else {
                          // check if it is external
                          Function *callee = call->getCalledFunction();
                          if (callee && callee->isDeclaration()) {
                              bb_extcall++;
                          }
                      }
                  }

                  if (I.getMetadata("afl_edge_sanitizer") != NULL) {
                      bb_san++;
                  }

                  if (dyn_cast<LoadInst>(&I) || dyn_cast<StoreInst>(&I) ||
                      dyn_cast<AtomicRMWInst>(&I) || dyn_cast<AtomicCmpXchgInst>(&I)) {
                      bb_memop++;
                  }
              }

              if (bb_cmp) {
                  Value* ArgCmp[] = {ConstantInt::get(Int32Ty, bb_cmp)};
                  IRB.CreateCall(AFLLogCmp, ArrayRef<Value*>(ArgCmp));
                  total_cmp += bb_cmp;
              }
              if (bb_indcall) {
                  Value* ArgIndCall[] = {ConstantInt::get(Int32Ty, bb_indcall)};
                  IRB.CreateCall(AFLLogIndCall, ArrayRef<Value*>(ArgIndCall));
                  total_indcall += bb_indcall;
              }
              if (bb_extcall) {
                  Value* ArgExtCall[] = {ConstantInt::get(Int32Ty, bb_extcall)};
                  IRB.CreateCall(AFLLogExtCall, ArrayRef<Value*>(ArgExtCall));
                  total_extcall += bb_extcall;
              }
              if (bb_san) {
                  /* Log covered label */
                  Value* ArgExtCall[] = {ConstantInt::get(Int32Ty, bb_san)};
                  IRB.CreateCall(AFLLogLabel, ArrayRef<Value*>(ArgExtCall));
                  total_san += bb_san;
              }
              if (bb_memop) {
                  Value* ArgExtCall[] = {ConstantInt::get(Int32Ty, bb_memop)};
                  IRB.CreateCall(AFLLogMemOp, ArrayRef<Value*>(ArgExtCall));
                  total_memop += bb_memop;
              }
          }
      }
      OKF("Collected %d cmp", total_cmp);
      OKF("Collected %d indirect call", total_indcall);
      OKF("Collected %d external all", total_extcall);
      OKF("Collected %d memops", total_memop);
      OKF("Collected %d sanitizer instrumentations", total_san);
  }

  /*log triggered sanitizer conditional block*/
  if (!annotate_for_se) {
    for (auto &F : M)
      for (auto &BB : F) {
        BasicBlock::iterator IP = BB.getFirstInsertionPt();
        IRBuilder<> IRB(&(*IP));
        if (is_sanitizer_handler(BB)) {
          /* Log block id of triggered bugs */
          BasicBlock* pred = BB.getSinglePredecessor();
          if (!pred) {
            WARNF("Sanitizer handler do not have single predecessor.");
          }else{
            int blk_id = get_block_id(*pred);
            // OKF("parent id: %d", blk_id);
            Value* ArgTriggered[] = {ConstantInt::get(Int32Ty, blk_id)};
            IRB.CreateCall(AFLLogTriggered, ArrayRef<Value*>(ArgTriggered));
          }
        }
      }
  }



  /* Say something nice. */

  if (!inst_blocks) WARNF("No instrumentation targets found.");
  else OKF("Instrumented %u locations (%s mode, ratio %u%%).",
           inst_blocks,
           ((getenv("AFL_USE_ASAN") || getenv("AFL_USE_MSAN")) ?
            "ASAN/MSAN" : "non-hardened"), inst_ratio);

  locMap.close();
  labelMap.close();
  return true;

}


static void registerInstrPass(const PassManagerBuilder &,
                            legacy::PassManagerBase &PM) {

  PM.add(new DynInstr());

}


static RegisterStandardPasses RegisterInstrPass(
                                              PassManagerBuilder::EP_OptimizerLast, registerInstrPass);

static RegisterStandardPasses RegisterInstrPass0(
                                               PassManagerBuilder::EP_EnabledOnOptLevel0, registerInstrPass);
