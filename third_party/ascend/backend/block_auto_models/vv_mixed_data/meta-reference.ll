define dso_local void @_mlir_ciface_cast_int32_t_to_int64_t(ptr nocapture noundef writeonly %sret, <64 x i32> noundef %src, <256 x i1> noundef %preg) local_unnamed_addr #38 !dbg !777006 {
entry:
  store <64 x i32> %src, ptr %sret, align 256, !dbg !777007, !tbaa !3423
  %0 = tail call <256 x i1> @llvm.hivm.pset.b32(i32 0), !dbg !777010
  %1 = tail call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %src, i16 31, <256 x i1> %0), !dbg !777007
  %arrayidx2.i = getelementptr inbounds [2 x <64 x i32>], ptr %sret, i64 0, i64 1, !dbg !777007
  store <64 x i32> %1, ptr %arrayidx2.i, align 256, !dbg !777007, !tbaa !3423
  ret void, !dbg !777012
}

define dso_local void @_mlir_ciface_vmuls_int64_t(ptr nocapture noundef writeonly %sret, ptr nocapture noundef readonly %src0, i64 noundef %src1, <256 x i1> noundef %preg) local_unnamed_addr #40 !dbg !775384 {
entry:
  %agg.tmp.sroa.0.sroa.0.0.copyload = load <64 x i32>, ptr %src0, align 256, !dbg !775385, !tbaa.struct !462224
  %agg.tmp.sroa.0.sroa.2.0.src0.sroa_idx = getelementptr inbounds i8, ptr %src0, i64 256, !dbg !775385
  %agg.tmp.sroa.0.sroa.2.0.copyload = load <64 x i32>, ptr %agg.tmp.sroa.0.sroa.2.0.src0.sroa_idx, align 256, !dbg !775385, !tbaa.struct !462225
  %conv.i.i = trunc i64 %src1 to i32, !dbg !775386
  %0 = tail call <64 x i32> @llvm.hivm.vbr.signed.v64i32(i32 %conv.i.i), !dbg !775386
  %1 = lshr i64 %src1, 32, !dbg !775386
  %conv2.i.i = trunc i64 %1 to i32, !dbg !775386
  %2 = tail call <64 x i32> @llvm.hivm.vbr.signed.v64i32(i32 %conv2.i.i), !dbg !775386
  %vmull.i.i = tail call { <64 x i32>, <64 x i32> } @llvm.hivm.vmull.v64u32(<64 x i32> %agg.tmp.sroa.0.sroa.0.0.copyload, <64 x i32> %0, <256 x i1> %preg), !dbg !775390
  %vmull.fca.0.extract.i.i = extractvalue { <64 x i32>, <64 x i32> } %vmull.i.i, 0, !dbg !775390
  %vmull.fca.1.extract.i.i = extractvalue { <64 x i32>, <64 x i32> } %vmull.i.i, 1, !dbg !775390
  %3 = tail call <64 x i32> @llvm.hivm.vmula.s.m.v64i32(<64 x i32> %vmull.fca.1.extract.i.i, <64 x i32> %agg.tmp.sroa.0.sroa.0.0.copyload, <64 x i32> %2, <256 x i1> %preg), !dbg !775390
  %4 = tail call <64 x i32> @llvm.hivm.vmula.s.m.v64i32(<64 x i32> %3, <64 x i32> %agg.tmp.sroa.0.sroa.2.0.copyload, <64 x i32> %0, <256 x i1> %preg), !dbg !775390
  store <64 x i32> %vmull.fca.0.extract.i.i, ptr %sret, align 256, !dbg !775390, !tbaa !3423
  %arrayidx26.i.i = getelementptr inbounds [2 x <64 x i32>], ptr %sret, i64 0, i64 1, !dbg !775390
  store <64 x i32> %4, ptr %arrayidx26.i.i, align 256, !dbg !775390, !tbaa !3423
  ret void, !dbg !775385
}

define dso_local void @_mlir_ciface_vbr_int64_t(ptr nocapture noundef writeonly %sret, i64 noundef %src) local_unnamed_addr #45 !dbg !777082 {
entry:
  %conv.i = trunc i64 %src to i32, !dbg !777083
  %0 = tail call <64 x i32> @llvm.hivm.vbr.signed.v64i32(i32 %conv.i), !dbg !777083
  store <64 x i32> %0, ptr %sret, align 256, !dbg !777083, !tbaa !3423
  %1 = lshr i64 %src, 32, !dbg !777083
  %conv2.i = trunc i64 %1 to i32, !dbg !777083
  %2 = tail call <64 x i32> @llvm.hivm.vbr.signed.v64i32(i32 %conv2.i), !dbg !777083
  %arrayidx6.i = getelementptr inbounds [2 x <64 x i32>], ptr %sret, i64 0, i64 1, !dbg !777083
  store <64 x i32> %2, ptr %arrayidx6.i, align 256, !dbg !777083, !tbaa !3423
  ret void, !dbg !777085
}

define dso_local void @_mlir_ciface_vadd_int64_t(ptr nocapture noundef writeonly %sret, ptr nocapture noundef readonly %src0, ptr nocapture noundef readonly %src1, <256 x i1> noundef %preg) local_unnamed_addr #40 !dbg !775187 {
entry:
  %agg.tmp.sroa.0.sroa.0.0.copyload = load <64 x i32>, ptr %src0, align 256, !dbg !775188, !tbaa.struct !462224
  %agg.tmp.sroa.0.sroa.2.0.src0.sroa_idx = getelementptr inbounds i8, ptr %src0, i64 256, !dbg !775188
  %agg.tmp.sroa.0.sroa.2.0.copyload = load <64 x i32>, ptr %agg.tmp.sroa.0.sroa.2.0.src0.sroa_idx, align 256, !dbg !775188, !tbaa.struct !462225
  %agg.tmp1.sroa.0.sroa.0.0.copyload = load <64 x i32>, ptr %src1, align 256, !dbg !775188, !tbaa.struct !462224
  %agg.tmp1.sroa.0.sroa.2.0.src1.sroa_idx = getelementptr inbounds i8, ptr %src1, i64 256, !dbg !775188
  %agg.tmp1.sroa.0.sroa.2.0.copyload = load <64 x i32>, ptr %agg.tmp1.sroa.0.sroa.2.0.src1.sroa_idx, align 256, !dbg !775188, !tbaa.struct !462225
  %0 = tail call { <64 x i32>, <256 x i1> } @llvm.hivm.vaddc.v64s32(<64 x i32> %agg.tmp.sroa.0.sroa.0.0.copyload, <64 x i32> %agg.tmp1.sroa.0.sroa.0.0.copyload, <256 x i1> %preg), !dbg !775189
  %1 = extractvalue { <64 x i32>, <256 x i1> } %0, 0, !dbg !775189
  %2 = extractvalue { <64 x i32>, <256 x i1> } %0, 1, !dbg !775189
  %3 = tail call { <64 x i32>, <256 x i1> } @llvm.hivm.vaddcs.v64s32(<64 x i32> %agg.tmp.sroa.0.sroa.2.0.copyload, <64 x i32> %agg.tmp1.sroa.0.sroa.2.0.copyload, <256 x i1> %2, <256 x i1> %preg), !dbg !775189
  %4 = extractvalue { <64 x i32>, <256 x i1> } %3, 0, !dbg !775189
  store <64 x i32> %1, ptr %sret, align 256, !dbg !775189, !tbaa !3423
  %arrayidx11.i = getelementptr inbounds [2 x <64 x i32>], ptr %sret, i64 0, i64 1, !dbg !775189
  store <64 x i32> %4, ptr %arrayidx11.i, align 256, !dbg !775189, !tbaa !3423
  ret void, !dbg !775188
}

define dso_local <256 x i1> @_mlir_ciface_vcmp_lt_int64_t(ptr nocapture noundef readonly %src0, ptr nocapture noundef readonly %src1, <256 x i1> noundef %preg) local_unnamed_addr #43 !dbg !775478 {
entry:
  %agg.tmp.sroa.0.sroa.0.0.copyload = load <64 x i32>, ptr %src0, align 256, !dbg !775479, !tbaa.struct !462224
  %agg.tmp.sroa.0.sroa.2.0.src0.sroa_idx = getelementptr inbounds i8, ptr %src0, i64 256, !dbg !775479
  %agg.tmp.sroa.0.sroa.2.0.copyload = load <64 x i32>, ptr %agg.tmp.sroa.0.sroa.2.0.src0.sroa_idx, align 256, !dbg !775479, !tbaa.struct !462225
  %agg.tmp1.sroa.0.sroa.0.0.copyload = load <64 x i32>, ptr %src1, align 256, !dbg !775479, !tbaa.struct !462224
  %agg.tmp1.sroa.0.sroa.2.0.src1.sroa_idx = getelementptr inbounds i8, ptr %src1, i64 256, !dbg !775479
  %agg.tmp1.sroa.0.sroa.2.0.copyload = load <64 x i32>, ptr %agg.tmp1.sroa.0.sroa.2.0.src1.sroa_idx, align 256, !dbg !775479, !tbaa.struct !462225
  %0 = tail call <256 x i1> @llvm.hivm.vcmp.lt.u.z.v64i32(<64 x i32> %agg.tmp.sroa.0.sroa.0.0.copyload, <64 x i32> %agg.tmp1.sroa.0.sroa.0.0.copyload, <256 x i1> %preg), !dbg !775480
  %1 = tail call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %agg.tmp.sroa.0.sroa.2.0.copyload, <64 x i32> %agg.tmp1.sroa.0.sroa.2.0.copyload, <256 x i1> %preg), !dbg !775480
  %2 = tail call <256 x i1> @llvm.hivm.vcmp.eq.s.z.v64i32(<64 x i32> %agg.tmp.sroa.0.sroa.2.0.copyload, <64 x i32> %agg.tmp1.sroa.0.sroa.2.0.copyload, <256 x i1> %preg), !dbg !775480
  %3 = tail call <256 x i1> @llvm.hivm.psel(<256 x i1> %0, <256 x i1> %1, <256 x i1> %2), !dbg !775480
  ret <256 x i1> %3, !dbg !775479
}

define dso_local void @_mlir_ciface_vdups_int64_t(ptr nocapture noundef writeonly %sret, i64 noundef %src, <256 x i1> noundef %preg) local_unnamed_addr #46 !dbg !777086 {
entry:
  %conv.i = trunc i64 %src to i32, !dbg !777087
  %0 = tail call <64 x i32> @llvm.hivm.vbr.signed.v64i32(i32 %conv.i), !dbg !777087
  store <64 x i32> %0, ptr %sret, align 256, !dbg !777087, !tbaa !3423
  %1 = lshr i64 %src, 32, !dbg !777087
  %conv2.i = trunc i64 %1 to i32, !dbg !777087
  %2 = tail call <64 x i32> @llvm.hivm.vbr.signed.v64i32(i32 %conv2.i), !dbg !777087
  %arrayidx6.i = getelementptr inbounds [2 x <64 x i32>], ptr %sret, i64 0, i64 1, !dbg !777087
  store <64 x i32> %2, ptr %arrayidx6.i, align 256, !dbg !777087, !tbaa !3423
  ret void, !dbg !777089
}

define dso_local void @_mlir_ciface_vload_NORM_int64_t_rank1(ptr nocapture noundef writeonly %sret, ptr nocapture noundef readonly %base, i64 noundef %offset) local_unnamed_addr #44 !dbg !775792 {
entry:
  %aligned.i = getelementptr inbounds %struct.memref_t.95, ptr %base, i64 0, i32 1, !dbg !775793
  %0 = load ptr addrspace(6), ptr %aligned.i, align 8, !dbg !775793, !tbaa !177809
  %offset1.i = getelementptr inbounds %struct.memref_t.95, ptr %base, i64 0, i32 2, !dbg !775796
  %1 = load i64, ptr %offset1.i, align 8, !dbg !775796, !tbaa !177812
  %add.ptr.i = getelementptr inbounds i64, ptr addrspace(6) %0, i64 %1, !dbg !775797
  %add.ptr2.i = getelementptr inbounds i64, ptr addrspace(6) %add.ptr.i, i64 %offset, !dbg !775798
  %vldsx2_v64s32.i.i = tail call { <64 x i32>, <64 x i32> } @llvm.hivm.vldsx2.v64i32(ptr addrspace(6) %add.ptr2.i, i32 0, i32 19, i32 0), !dbg !775799
  %vldsx2_v64s32.i.i.elt = extractvalue { <64 x i32>, <64 x i32> } %vldsx2_v64s32.i.i, 0, !dbg !775799
  store <64 x i32> %vldsx2_v64s32.i.i.elt, ptr %sret, align 256, !dbg !775799
  %sret.repack1 = getelementptr inbounds { <64 x i32>, <64 x i32> }, ptr %sret, i64 0, i32 1, !dbg !775799
  %vldsx2_v64s32.i.i.elt2 = extractvalue { <64 x i32>, <64 x i32> } %vldsx2_v64s32.i.i, 1, !dbg !775799
  store <64 x i32> %vldsx2_v64s32.i.i.elt2, ptr %sret.repack1, align 256, !dbg !775799
  ret void, !dbg !775801
}

define dso_local void @_mlir_ciface_masked_store_NORM_B64_int64_t_rank1(ptr nocapture noundef readonly %base, i64 noundef %offset, <256 x i1> noundef %preg, ptr nocapture noundef readonly %src) local_unnamed_addr #8 !dbg !776216 {
entry:
  %aligned.i = getelementptr inbounds %struct.memref_t.95, ptr %base, i64 0, i32 1, !dbg !776217
  %0 = load ptr addrspace(6), ptr %aligned.i, align 8, !dbg !776217, !tbaa !177809
  %offset1.i = getelementptr inbounds %struct.memref_t.95, ptr %base, i64 0, i32 2, !dbg !776220
  %1 = load i64, ptr %offset1.i, align 8, !dbg !776220, !tbaa !177812
  %add.ptr.i = getelementptr inbounds i64, ptr addrspace(6) %0, i64 %1, !dbg !776221
  %add.ptr2.i = getelementptr inbounds i64, ptr addrspace(6) %add.ptr.i, i64 %offset, !dbg !776222
  %agg.tmp.i.sroa.0.sroa.0.0.copyload = load <64 x i32>, ptr %src, align 256, !dbg !776223, !tbaa.struct !462224
  %agg.tmp.i.sroa.0.sroa.4.0.src.sroa_idx = getelementptr inbounds i8, ptr %src, i64 256, !dbg !776223
  %agg.tmp.i.sroa.0.sroa.4.0.copyload = load <64 x i32>, ptr %agg.tmp.i.sroa.0.sroa.4.0.src.sroa_idx, align 256, !dbg !776223, !tbaa.struct !462225
  %pintlv.i.i = tail call { <256 x i1>, <256 x i1> } @llvm.hivm.pintlv.b32(<256 x i1> %preg, <256 x i1> %preg), !dbg !776224
  %2 = extractvalue { <256 x i1>, <256 x i1> } %pintlv.i.i, 0, !dbg !776224
  %3 = extractvalue { <256 x i1>, <256 x i1> } %pintlv.i.i, 1, !dbg !776224
  %vintlv.i.i.i = tail call { <64 x i32>, <64 x i32> } @llvm.hivm.vintlv.v64i32(<64 x i32> %agg.tmp.i.sroa.0.sroa.0.0.copyload, <64 x i32> %agg.tmp.i.sroa.0.sroa.4.0.copyload), !dbg !776227
  %vintlv.fca.0.extract.i.i.i = extractvalue { <64 x i32>, <64 x i32> } %vintlv.i.i.i, 0, !dbg !776227
  %vintlv.fca.1.extract.i.i.i = extractvalue { <64 x i32>, <64 x i32> } %vintlv.i.i.i, 1, !dbg !776227
  tail call void @llvm.hivm.vstsx1.v64i32(<64 x i32> %vintlv.fca.0.extract.i.i.i, ptr addrspace(6) %add.ptr2.i, i32 0, i32 2, i32 0, <256 x i1> %2), !dbg !776229
  tail call void @llvm.hivm.vstsx1.v64i32(<64 x i32> %vintlv.fca.1.extract.i.i.i, ptr addrspace(6) %add.ptr2.i, i32 256, i32 2, i32 0, <256 x i1> %3), !dbg !776231
  ret void, !dbg !776233
}

define dso_local void @_mlir_ciface_indirect_load_nonvolatile_1d_bfloat16_t_int64_t(ptr nocapture noundef readonly %src, ptr nocapture noundef readonly %idx, ptr nocapture noundef readonly %dst, ptr nocapture noundef readonly %mask, ptr nocapture noundef readonly %other) local_unnamed_addr #16 !dbg !698961 {
entry:
  %aligned.i = getelementptr inbounds %struct.memref_t, ptr %src, i64 0, i32 1, !dbg !698962
  %0 = load ptr addrspace(1), ptr %aligned.i, align 8, !dbg !698962, !tbaa !96398
  %aligned2.i = getelementptr inbounds %struct.memref_t.95, ptr %idx, i64 0, i32 1, !dbg !698965
  %1 = load ptr addrspace(6), ptr %aligned2.i, align 8, !dbg !698965, !tbaa !177809
  %offset.i = getelementptr inbounds %struct.memref_t.95, ptr %idx, i64 0, i32 2, !dbg !698966
  %2 = load i64, ptr %offset.i, align 8, !dbg !698966, !tbaa !177812
  %add.ptr.i = getelementptr inbounds i64, ptr addrspace(6) %1, i64 %2, !dbg !698967
  %aligned4.i = getelementptr inbounds %struct.memref_t.95, ptr %dst, i64 0, i32 1, !dbg !698968
  %3 = load ptr addrspace(6), ptr %aligned4.i, align 8, !dbg !698968, !tbaa !203431
  %offset5.i = getelementptr inbounds %struct.memref_t.95, ptr %dst, i64 0, i32 2, !dbg !698969
  %4 = load i64, ptr %offset5.i, align 8, !dbg !698969, !tbaa !203434
  %add.ptr6.i = getelementptr inbounds bfloat, ptr addrspace(6) %3, i64 %4, !dbg !698970
  %aligned8.i = getelementptr inbounds %struct.memref_t.95, ptr %mask, i64 0, i32 1, !dbg !698971
  %5 = load ptr addrspace(6), ptr %aligned8.i, align 8, !dbg !698971, !tbaa !220570
  %offset9.i = getelementptr inbounds %struct.memref_t.95, ptr %mask, i64 0, i32 2, !dbg !698972
  %6 = load i64, ptr %offset9.i, align 8, !dbg !698972, !tbaa !220573
  %add.ptr10.i = getelementptr inbounds i8, ptr addrspace(6) %5, i64 %6, !dbg !698973
  %aligned12.i = getelementptr inbounds %struct.memref_t.95, ptr %other, i64 0, i32 1, !dbg !698974
  %7 = load ptr addrspace(6), ptr %aligned12.i, align 8, !dbg !698974, !tbaa !203431
  %offset13.i = getelementptr inbounds %struct.memref_t.95, ptr %other, i64 0, i32 2, !dbg !698975
  %8 = load i64, ptr %offset13.i, align 8, !dbg !698975, !tbaa !203434
  %add.ptr14.i = getelementptr inbounds bfloat, ptr addrspace(6) %7, i64 %8, !dbg !698976
  %sizes.i = getelementptr inbounds %struct.memref_t.95, ptr %idx, i64 0, i32 3, !dbg !698977
  %strides.i = getelementptr inbounds %struct.memref_t.95, ptr %idx, i64 0, i32 4, !dbg !698978
  %strides16.i = getelementptr inbounds %struct.memref_t.95, ptr %dst, i64 0, i32 4, !dbg !698979
  %strides18.i = getelementptr inbounds %struct.memref_t.95, ptr %mask, i64 0, i32 4, !dbg !698980
  %strides20.i = getelementptr inbounds %struct.memref_t.95, ptr %other, i64 0, i32 4, !dbg !698981
  tail call void @llvm.hivm.store.vfsimt.info(i64 4295033856), !dbg !698982
  %9 = load i64, ptr %sizes.i, align 8, !dbg !698985, !tbaa !3381
  %conv30.i = trunc i64 %9 to i32, !dbg !698985
  %10 = load i64, ptr %strides.i, align 8, !dbg !698985, !tbaa !3381
  %conv31.i = trunc i64 %10 to i32, !dbg !698985
  %11 = load i64, ptr %strides16.i, align 8, !dbg !698985, !tbaa !3381
  %conv32.i = trunc i64 %11 to i32, !dbg !698985
  %12 = load i64, ptr %strides18.i, align 8, !dbg !698985, !tbaa !3381
  %conv33.i = trunc i64 %12 to i32, !dbg !698985
  %13 = load i64, ptr %strides20.i, align 8, !dbg !698985, !tbaa !3381
  %conv34.i = trunc i64 %13 to i32, !dbg !698985
  tail call simt_entry void @_ZL18simtIndirectLoad1DIu6__bf16lLb0EEvN18IndirectLoadSrcPtrIT_XT1_EE4typeEPU3AS6T0_PU3AS6S1_PU3AS6bS8_jjjjj_simt_entry(ptr addrspace(1) noundef %0, ptr addrspace(6) noundef %add.ptr.i, ptr addrspace(6) noundef %add.ptr6.i, ptr addrspace(6) noundef %add.ptr10.i, ptr addrspace(6) noundef %add.ptr14.i, i32 noundef %conv30.i, i32 noundef %conv31.i, i32 noundef %conv32.i, i32 noundef %conv33.i, i32 noundef %conv34.i), !dbg !698986
  ret void, !dbg !698987
}

define internal simt_entry void @_ZL18simtIndirectLoad1DIu6__bf16lLb0EEvN18IndirectLoadSrcPtrIT_XT1_EE4typeEPU3AS6T0_PU3AS6S1_PU3AS6bS8_jjjjj_simt_entry(ptr addrspace(1) nocapture noundef readonly %0, ptr addrspace(6) nocapture noundef readonly %1, ptr addrspace(6) nocapture noundef writeonly %2, ptr addrspace(6) nocapture noundef readonly %3, ptr addrspace(6) nocapture noundef readonly %4, i32 noundef %5, i32 noundef %6, i32 noundef %7, i32 noundef %8, i32 noundef %9) unnamed_addr #29 !dbg !698988 !annotation !602984 !annotation !602985 !annotation !602984 !annotation !602985 {
entry:
  %10 = tail call i32 @llvm.hivm.get.TID.X(), !dbg !698989
  %cmp1 = icmp ult i32 %10, %5, !dbg !698991, !loop.guard !3420
  br i1 %cmp1, label %for.body.lr.ph, label %for.end, !dbg !698992, !loop.guard !3420

for.body.lr.ph:                                   ; preds = %entry
  %11 = tail call i32 @llvm.hivm.get.BLOCK.DIM.X(), !dbg !698993
  br label %for.body, !dbg !698992

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %i0.02 = phi i32 [ %10, %for.body.lr.ph ], [ %add, %for.inc ]
  %mul = mul i32 %i0.02, %8, !dbg !698994
  %mul1 = mul i32 %i0.02, %7, !dbg !698995
  %idxprom = zext i32 %mul to i64, !dbg !698996
  %arrayidx = getelementptr inbounds i8, ptr addrspace(6) %3, i64 %idxprom, !dbg !698996
  %12 = load i8, ptr addrspace(6) %arrayidx, align 1, !dbg !698996, !tbaa !697238, !range !697240
  %tobool.not = icmp eq i8 %12, 0, !dbg !698996
  br i1 %tobool.not, label %if.else, label %if.then, !dbg !698996

if.then:                                          ; preds = %for.body
  %mul2 = mul i32 %i0.02, %6, !dbg !698997
  %idxprom3 = zext i32 %mul2 to i64, !dbg !698998
  %arrayidx4 = getelementptr inbounds i64, ptr addrspace(6) %1, i64 %idxprom3, !dbg !698998
  %13 = load i64, ptr addrspace(6) %arrayidx4, align 8, !dbg !698998, !tbaa !3381
  %arrayidx5 = getelementptr inbounds bfloat, ptr addrspace(1) %0, i64 %13, !dbg !698999
  %14 = load i16, ptr addrspace(1) %arrayidx5, align 2, !dbg !698999, !tbaa !96570
  br label %for.inc, !dbg !699000

if.else:                                          ; preds = %for.body
  %mul8 = mul i32 %i0.02, %9, !dbg !699001
  %idxprom9 = zext i32 %mul8 to i64, !dbg !699002
  %arrayidx10 = getelementptr inbounds bfloat, ptr addrspace(6) %4, i64 %idxprom9, !dbg !699002
  %15 = load i16, ptr addrspace(6) %arrayidx10, align 2, !dbg !699002, !tbaa !96570
  br label %for.inc

for.inc:                                          ; preds = %if.else, %if.then
  %.sink = phi i16 [ %15, %if.else ], [ %14, %if.then ]
  %idxprom11 = zext i32 %mul1 to i64, !dbg !698993
  %arrayidx12 = getelementptr inbounds bfloat, ptr addrspace(6) %2, i64 %idxprom11, !dbg !698993
  store i16 %.sink, ptr addrspace(6) %arrayidx12, align 2, !dbg !698993, !tbaa !96570
  %add = add i32 %11, %i0.02, !dbg !699003
  %cmp = icmp ult i32 %add, %5, !dbg !698991
  br i1 %cmp, label %for.body, label %for.end, !dbg !698992, !llvm.loop !699004

for.end:                                          ; preds = %for.inc, %entry
  ret void, !dbg !699006
}
