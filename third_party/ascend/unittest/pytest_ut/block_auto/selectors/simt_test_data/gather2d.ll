define private fastcc void @indirect_load_nonvolatile_2d_float_int64_t(ptr addrspace(1) %0) unnamed_addr #4 {
if.else.i:
  tail call void @llvm.hivm.store.vfsimt.info(i64 4295491592), !dbg !966
  tail call simt_entry void @_ZL18simtIndirectLoad2DIflLb0EEvN18IndirectLoadSrcPtrIT_XT1_EE4typeEPU3AS6T0_PU3AS6S1_PU3AS6bS8_jjjjjjjjjj_simt_entry(ptr addrspace(1) noundef %0), !dbg !974
  ret void
}

define internal simt_entry void @_ZL18simtIndirectLoad2DIflLb0EEvN18IndirectLoadSrcPtrIT_XT1_EE4typeEPU3AS6T0_PU3AS6S1_PU3AS6bS8_jjjjjjjjjj_simt_entry(ptr addrspace(1) noundef %0) unnamed_addr #9 !dbg !1063 !annotation !1065 !annotation !1066 !annotation !1065 !annotation !1066 {
entry:
  %1 = tail call i32 @llvm.hivm.get.TID.Y(), !dbg !1067
  %cmp4 = icmp ult i32 %1, 8, !dbg !1071, !loop.guard !993
  br i1 %cmp4, label %for.body.lr.ph, label %for.end30, !dbg !1072, !loop.guard !993

for.body.lr.ph:                                   ; preds = %entry
  %2 = tail call i32 @llvm.hivm.get.TID.X(), !dbg !1073
  %cmp31 = icmp ult i32 %2, 8, !loop.guard !993
  %3 = tail call i32 @llvm.hivm.get.BLOCK.DIM.Y(), !dbg !1073
  br label %for.body, !dbg !1072

for.body:                                         ; preds = %for.body.lr.ph, %for.inc27
  %ty.05 = phi i32 [ %1, %for.body.lr.ph ], [ %add29, %for.inc27 ]
  br i1 %cmp31, label %for.body5.lr.ph, label %for.inc27, !dbg !1074, !loop.guard !993

for.body5.lr.ph:                                  ; preds = %for.body
  %mul = shl nuw nsw i32 %ty.05, 3
  %4 = tail call i32 @llvm.hivm.get.BLOCK.DIM.X(), !dbg !1073
  br label %for.body5, !dbg !1074

for.body5:                                        ; preds = %for.body5.lr.ph, %for.inc
  %tx.02 = phi i32 [ %2, %for.body5.lr.ph ], [ %add26, %for.inc ]
  %add = add nuw i32 %tx.02, %mul, !dbg !1075
  %idxprom = zext i32 %add to i64, !dbg !1076
  %arrayidx = getelementptr inbounds i8, ptr addrspace(6) inttoptr (i64 832 to ptr addrspace(6)), i64 %idxprom, !dbg !1076
  %5 = load i8, ptr addrspace(6) %arrayidx, align 1, !dbg !1076, !tbaa !1077, !range !1079
  %tobool.not = icmp eq i8 %5, 0, !dbg !1076
  br i1 %tobool.not, label %if.else, label %if.then, !dbg !1076

if.then:                                          ; preds = %for.body5
  %arrayidx14 = getelementptr inbounds i64, ptr addrspace(6) inttoptr (i64 320 to ptr addrspace(6)), i64 %idxprom, !dbg !1080
  %6 = load i64, ptr addrspace(6) %arrayidx14, align 8, !dbg !1080, !tbaa !1081
  %arrayidx15 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %6, !dbg !1083
  %7 = load i32, ptr addrspace(1) %arrayidx15, align 4, !dbg !1083, !tbaa !998
  br label %for.inc, !dbg !1084

if.else:                                          ; preds = %for.body5
  %arrayidx22 = getelementptr inbounds float, ptr addrspace(6) null, i64 %idxprom, !dbg !1085
  %8 = load i32, ptr addrspace(6) %arrayidx22, align 4, !dbg !1085, !tbaa !998
  br label %for.inc

for.inc:                                          ; preds = %if.else, %if.then
  %.sink = phi i32 [ %8, %if.else ], [ %7, %if.then ], !dbg !1073
  %9 = getelementptr inbounds float, ptr addrspace(6) inttoptr (i64 896 to ptr addrspace(6)), i64 %idxprom, !dbg !1086
  store i32 %.sink, ptr addrspace(6) %9, align 4, !dbg !1087
  %add26 = add i32 %4, %tx.02, !dbg !1088
  %cmp3 = icmp ult i32 %add26, 8, !dbg !1089
  br i1 %cmp3, label %for.body5, label %for.inc27, !dbg !1074, !llvm.loop !1090

for.inc27:                                        ; preds = %for.inc, %for.body
  %add29 = add i32 %3, %ty.05, !dbg !1092
  %cmp = icmp ult i32 %add29, 8, !dbg !1071
  br i1 %cmp, label %for.body, label %for.end30, !dbg !1072, !llvm.loop !1093

for.end30:                                        ; preds = %for.inc27, %entry
  ret void, !dbg !1095
}
