define private fastcc void @cumsum_1d_float_dim0() unnamed_addr #6 {
if.end10.i:
  tail call void @llvm.hivm.BARRIER(i64 6), !dbg !1022
  tail call void @llvm.hivm.store.vfsimt.info(i64 4296015904), !dbg !1027
  tail call simt_entry void @_ZL37simt_sklansky_scan_1d_block_unroll_p2IfEvPU3AS6T_S2_iiS2__simt_entry(), !dbg !1031
  tail call void @llvm.hivm.BARRIER(i64 6), !dbg !1032
  ret void
}

define internal simt_entry void @_ZL37simt_sklansky_scan_1d_block_unroll_p2IfEvPU3AS6T_S2_iiS2__simt_entry() unnamed_addr #12 !dbg !1051 !annotation !1053 !annotation !1054 !annotation !1053 !annotation !1054 {
entry:
  %0 = tail call i32 @llvm.hivm.get.TID.X(), !dbg !1055
  %1 = tail call i32 @llvm.hivm.get.TID.Y(), !dbg !1059
  %2 = tail call i32 @llvm.hivm.get.BLOCK.DIM.Y(), !dbg !1062
  %mul = shl nsw i32 %1, 5, !dbg !1065
  %add = add nsw i32 %mul, %0, !dbg !1066
  %cmp = icmp slt i32 %add, 512, !dbg !1067
  br i1 %cmp, label %cond.true5, label %cond.end7, !dbg !1068

cond.true5:                                       ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !1069
  %arrayidx = getelementptr inbounds float, ptr addrspace(6) null, i64 %idxprom, !dbg !1069
  %3 = load float, ptr addrspace(6) %arrayidx, align 4, !dbg !1069, !tbaa !940
  br label %cond.end7, !dbg !1068

cond.end7:                                        ; preds = %cond.true5, %entry
  %cond8 = phi float [ %3, %cond.true5 ], [ 0.000000e+00, %entry ], !dbg !1068
  %and = and i32 %0, -7938, !dbg !1070
  %4 = bitcast float %cond8 to i32, !dbg !1071
  %or1.i = or i32 %and, 7936, !dbg !1074
  %5 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %4, i32 %or1.i), !dbg !1074
  %6 = bitcast i32 %5 to float, !dbg !1078
  %and15 = and i32 %0, 1, !dbg !1079
  %cmp16.not.not.not = icmp eq i32 %and15, 0, !dbg !1080
  %add17 = select i1 %cmp16.not.not.not, float -0.000000e+00, float %6, !dbg !1081
  %spec.select = fadd float %cond8, %add17, !dbg !1081
  %and.1 = and i32 %0, -7940, !dbg !1070
  %7 = bitcast float %spec.select to i32, !dbg !1071
  %or1.i.1 = or i32 %and.1, 7937, !dbg !1074
  %8 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %7, i32 %or1.i.1), !dbg !1074
  %9 = bitcast i32 %8 to float, !dbg !1078
  %and15.1 = and i32 %0, 2, !dbg !1079
  %cmp16.not.1.not.not = icmp eq i32 %and15.1, 0, !dbg !1080
  %add17.1 = select i1 %cmp16.not.1.not.not, float -0.000000e+00, float %9, !dbg !1081
  %spec.select.1 = fadd float %spec.select, %add17.1, !dbg !1081
  %and.2 = and i32 %0, -7944, !dbg !1070
  %10 = bitcast float %spec.select.1 to i32, !dbg !1071
  %or1.i.2 = or i32 %and.2, 7939, !dbg !1074
  %11 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %10, i32 %or1.i.2), !dbg !1074
  %12 = bitcast i32 %11 to float, !dbg !1078
  %and15.2 = and i32 %0, 4, !dbg !1079
  %cmp16.not.2.not.not = icmp eq i32 %and15.2, 0, !dbg !1080
  %add17.2 = select i1 %cmp16.not.2.not.not, float -0.000000e+00, float %12, !dbg !1081
  %spec.select.2 = fadd float %spec.select.1, %add17.2, !dbg !1081
  %and.3 = and i32 %0, -7952, !dbg !1070
  %13 = bitcast float %spec.select.2 to i32, !dbg !1071
  %or1.i.3 = or i32 %and.3, 7943, !dbg !1074
  %14 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %13, i32 %or1.i.3), !dbg !1074
  %15 = bitcast i32 %14 to float, !dbg !1078
  %and15.3 = and i32 %0, 8, !dbg !1079
  %cmp16.not.3.not.not = icmp eq i32 %and15.3, 0, !dbg !1080
  %add17.3 = select i1 %cmp16.not.3.not.not, float -0.000000e+00, float %15, !dbg !1081
  %spec.select.3 = fadd float %spec.select.2, %add17.3, !dbg !1081
  %and.4 = and i32 %0, -7968, !dbg !1070
  %16 = bitcast float %spec.select.3 to i32, !dbg !1071
  %or1.i.4 = or i32 %and.4, 7951, !dbg !1074
  %17 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %16, i32 %or1.i.4), !dbg !1074
  %18 = bitcast i32 %17 to float, !dbg !1078
  %and15.4 = and i32 %0, 16, !dbg !1079
  %cmp16.not.4.not.not = icmp eq i32 %and15.4, 0, !dbg !1080
  %add17.4 = select i1 %cmp16.not.4.not.not, float -0.000000e+00, float %18, !dbg !1081
  %spec.select.4 = fadd float %spec.select.3, %add17.4, !dbg !1081
  %19 = bitcast float %spec.select.4 to i32, !dbg !1082
  %20 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %19, i32 7967), !dbg !1084
  %cmp28 = icmp eq i32 %0, 0, !dbg !1086
  br i1 %cmp28, label %if.then29, label %if.end32, !dbg !1087

if.then29:                                        ; preds = %cond.end7
  %idxprom30 = sext i32 %1 to i64, !dbg !1088
  %arrayidx31 = getelementptr inbounds float, ptr addrspace(6) inttoptr (i64 6176 to ptr addrspace(6)), i64 %idxprom30, !dbg !1088
  store i32 %20, ptr addrspace(6) %arrayidx31, align 4, !dbg !1089, !tbaa !940
  br label %if.end32, !dbg !1090

if.end32:                                         ; preds = %if.then29, %cond.end7
  tail call void @llvm.hivm.sync.workitems(), !dbg !1091
  %cmp33 = icmp slt i32 %0, %2, !dbg !1094
  br i1 %cmp33, label %cond.true34, label %cond.end38, !dbg !1095

cond.true34:                                      ; preds = %if.end32
  %idxprom35 = sext i32 %0 to i64, !dbg !1096
  %arrayidx36 = getelementptr inbounds float, ptr addrspace(6) inttoptr (i64 6176 to ptr addrspace(6)), i64 %idxprom35, !dbg !1096
  %21 = load float, ptr addrspace(6) %arrayidx36, align 4, !dbg !1096, !tbaa !940
  br label %cond.end38, !dbg !1095

cond.end38:                                       ; preds = %cond.true34, %if.end32
  %cond39 = phi float [ %21, %cond.true34 ], [ 0.000000e+00, %if.end32 ], !dbg !1095
  %cmp40 = icmp sgt i32 %2, 1, !dbg !1097
  br i1 %cmp40, label %if.then41, label %if.end62, !dbg !1098

if.then41:                                        ; preds = %cond.end38
  %and43 = and i32 %1, 1, !dbg !1099
  %cmp44.not = icmp eq i32 %and43, 0, !dbg !1100
  br i1 %cmp44.not, label %if.then41.if.end51_crit_edge, label %if.then45, !dbg !1101

if.then41.if.end51_crit_edge:                     ; preds = %if.then41
  %.pre = bitcast float %cond39 to i32, !dbg !1102
  br label %if.end51, !dbg !1101

if.then45:                                        ; preds = %if.then41
  %and46 = and i32 %1, -7938, !dbg !1104
  %or1.i86 = or i32 %and46, 7936, !dbg !1105
  %22 = bitcast float %cond39 to i32, !dbg !1108
  %23 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %22, i32 %or1.i86), !dbg !1105
  %24 = bitcast i32 %23 to float, !dbg !1109
  %add50 = fadd float %spec.select.4, %24, !dbg !1110
  br label %if.end51, !dbg !1111

if.end51:                                         ; preds = %if.then41.if.end51_crit_edge, %if.then45
  %.pre-phi = phi i32 [ %.pre, %if.then41.if.end51_crit_edge ], [ %22, %if.then45 ], !dbg !1102
  %v.2 = phi float [ %spec.select.4, %if.then41.if.end51_crit_edge ], [ %add50, %if.then45 ], !dbg !1112
  %and52 = and i32 %0, -7938, !dbg !1113
  %or1.i87 = or i32 %and52, 7936, !dbg !1114
  %25 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %.pre-phi, i32 %or1.i87), !dbg !1114
  %26 = bitcast i32 %25 to float, !dbg !1116
  %cmp58.not = icmp eq i32 %and15, 0, !dbg !1117
  %add60 = select i1 %cmp58.not, float -0.000000e+00, float %26, !dbg !1118
  %spec.select70 = fadd float %cond39, %add60, !dbg !1118
  br label %if.end62, !dbg !1119

if.end62:                                         ; preds = %if.end51, %cond.end38
  %t.1 = phi float [ %spec.select70, %if.end51 ], [ %cond39, %cond.end38 ], !dbg !1112
  %v.3 = phi float [ %v.2, %if.end51 ], [ %spec.select.4, %cond.end38 ], !dbg !1112
  %cmp63 = icmp sgt i32 %2, 2, !dbg !1120
  br i1 %cmp63, label %if.then64, label %if.end88, !dbg !1121

if.then64:                                        ; preds = %if.end62
  %and66 = and i32 %1, 2, !dbg !1122
  %cmp67.not = icmp eq i32 %and66, 0, !dbg !1123
  br i1 %cmp67.not, label %if.then64.if.end76_crit_edge, label %if.then68, !dbg !1124

if.then64.if.end76_crit_edge:                     ; preds = %if.then64
  %.pre3 = bitcast float %t.1 to i32, !dbg !1125
  br label %if.end76, !dbg !1124

if.then68:                                        ; preds = %if.then64
  %and70 = and i32 %1, -7940, !dbg !1127
  %or1.i88 = or i32 %and70, 7937, !dbg !1128
  %27 = bitcast float %t.1 to i32, !dbg !1131
  %28 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %27, i32 %or1.i88), !dbg !1128
  %29 = bitcast i32 %28 to float, !dbg !1132
  %add75 = fadd float %v.3, %29, !dbg !1133
  br label %if.end76, !dbg !1134

if.end76:                                         ; preds = %if.then64.if.end76_crit_edge, %if.then68
  %.pre-phi4 = phi i32 [ %.pre3, %if.then64.if.end76_crit_edge ], [ %27, %if.then68 ], !dbg !1125
  %v.4 = phi float [ %v.3, %if.then64.if.end76_crit_edge ], [ %add75, %if.then68 ], !dbg !1112
  %and78 = and i32 %0, -7940, !dbg !1135
  %or1.i89 = or i32 %and78, 7937, !dbg !1136
  %30 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %.pre-phi4, i32 %or1.i89), !dbg !1136
  %31 = bitcast i32 %30 to float, !dbg !1138
  %and83 = and i32 %0, 2, !dbg !1139
  %cmp84.not = icmp eq i32 %and83, 0, !dbg !1140
  %add86 = select i1 %cmp84.not, float -0.000000e+00, float %31, !dbg !1141
  %spec.select71 = fadd float %t.1, %add86, !dbg !1141
  br label %if.end88, !dbg !1142

if.end88:                                         ; preds = %if.end76, %if.end62
  %t.3 = phi float [ %spec.select71, %if.end76 ], [ %t.1, %if.end62 ], !dbg !1112
  %v.5 = phi float [ %v.4, %if.end76 ], [ %v.3, %if.end62 ], !dbg !1112
  %cmp89 = icmp sgt i32 %2, 4, !dbg !1143
  br i1 %cmp89, label %if.then90, label %if.end114, !dbg !1144

if.then90:                                        ; preds = %if.end88
  %and92 = and i32 %1, 4, !dbg !1145
  %cmp93.not = icmp eq i32 %and92, 0, !dbg !1146
  br i1 %cmp93.not, label %if.then90.if.end102_crit_edge, label %if.then94, !dbg !1147

if.then90.if.end102_crit_edge:                    ; preds = %if.then90
  %.pre5 = bitcast float %t.3 to i32, !dbg !1148
  br label %if.end102, !dbg !1147

if.then94:                                        ; preds = %if.then90
  %and96 = and i32 %1, -7944, !dbg !1150
  %or1.i90 = or i32 %and96, 7939, !dbg !1151
  %32 = bitcast float %t.3 to i32, !dbg !1154
  %33 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %32, i32 %or1.i90), !dbg !1151
  %34 = bitcast i32 %33 to float, !dbg !1155
  %add101 = fadd float %v.5, %34, !dbg !1156
  br label %if.end102, !dbg !1157

if.end102:                                        ; preds = %if.then90.if.end102_crit_edge, %if.then94
  %.pre-phi6 = phi i32 [ %.pre5, %if.then90.if.end102_crit_edge ], [ %32, %if.then94 ], !dbg !1148
  %v.6 = phi float [ %v.5, %if.then90.if.end102_crit_edge ], [ %add101, %if.then94 ], !dbg !1112
  %and104 = and i32 %0, -7944, !dbg !1158
  %or1.i91 = or i32 %and104, 7939, !dbg !1159
  %35 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %.pre-phi6, i32 %or1.i91), !dbg !1159
  %36 = bitcast i32 %35 to float, !dbg !1161
  %and109 = and i32 %0, 4, !dbg !1162
  %cmp110.not = icmp eq i32 %and109, 0, !dbg !1163
  %add112 = select i1 %cmp110.not, float -0.000000e+00, float %36, !dbg !1164
  %spec.select72 = fadd float %t.3, %add112, !dbg !1164
  br label %if.end114, !dbg !1165

if.end114:                                        ; preds = %if.end102, %if.end88
  %t.5 = phi float [ %spec.select72, %if.end102 ], [ %t.3, %if.end88 ], !dbg !1112
  %v.7 = phi float [ %v.6, %if.end102 ], [ %v.5, %if.end88 ], !dbg !1112
  %cmp115 = icmp sgt i32 %2, 8, !dbg !1166
  br i1 %cmp115, label %if.then116, label %if.end140, !dbg !1167

if.then116:                                       ; preds = %if.end114
  %and118 = and i32 %1, 8, !dbg !1168
  %cmp119.not = icmp eq i32 %and118, 0, !dbg !1169
  br i1 %cmp119.not, label %if.then116.if.end128_crit_edge, label %if.then120, !dbg !1170

if.then116.if.end128_crit_edge:                   ; preds = %if.then116
  %.pre7 = bitcast float %t.5 to i32, !dbg !1171
  br label %if.end128, !dbg !1170

if.then120:                                       ; preds = %if.then116
  %and122 = and i32 %1, -7952, !dbg !1173
  %or1.i92 = or i32 %and122, 7943, !dbg !1174
  %37 = bitcast float %t.5 to i32, !dbg !1177
  %38 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %37, i32 %or1.i92), !dbg !1174
  %39 = bitcast i32 %38 to float, !dbg !1178
  %add127 = fadd float %v.7, %39, !dbg !1179
  br label %if.end128, !dbg !1180

if.end128:                                        ; preds = %if.then116.if.end128_crit_edge, %if.then120
  %.pre-phi8 = phi i32 [ %.pre7, %if.then116.if.end128_crit_edge ], [ %37, %if.then120 ], !dbg !1171
  %v.8 = phi float [ %v.7, %if.then116.if.end128_crit_edge ], [ %add127, %if.then120 ], !dbg !1112
  %and130 = and i32 %0, -7952, !dbg !1181
  %or1.i93 = or i32 %and130, 7943, !dbg !1182
  %40 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %.pre-phi8, i32 %or1.i93), !dbg !1182
  %41 = bitcast i32 %40 to float, !dbg !1184
  %and135 = and i32 %0, 8, !dbg !1185
  %cmp136.not = icmp eq i32 %and135, 0, !dbg !1186
  %add138 = select i1 %cmp136.not, float -0.000000e+00, float %41, !dbg !1187
  %spec.select73 = fadd float %t.5, %add138, !dbg !1187
  br label %if.end140, !dbg !1188

if.end140:                                        ; preds = %if.end128, %if.end114
  %t.7 = phi float [ %spec.select73, %if.end128 ], [ %t.5, %if.end114 ], !dbg !1112
  %v.9 = phi float [ %v.8, %if.end128 ], [ %v.7, %if.end114 ], !dbg !1112
  %cmp141 = icmp sgt i32 %2, 16, !dbg !1189
  br i1 %cmp141, label %if.then142, label %if.end166, !dbg !1190

if.then142:                                       ; preds = %if.end140
  %and144 = and i32 %1, 16, !dbg !1191
  %cmp145.not = icmp eq i32 %and144, 0, !dbg !1192
  br i1 %cmp145.not, label %if.then142.if.end154_crit_edge, label %if.then146, !dbg !1193

if.then142.if.end154_crit_edge:                   ; preds = %if.then142
  %.pre9 = bitcast float %t.7 to i32, !dbg !1194
  br label %if.end154, !dbg !1193

if.then146:                                       ; preds = %if.then142
  %and148 = and i32 %1, -7968, !dbg !1196
  %or1.i94 = or i32 %and148, 7951, !dbg !1197
  %42 = bitcast float %t.7 to i32, !dbg !1200
  %43 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %42, i32 %or1.i94), !dbg !1197
  %44 = bitcast i32 %43 to float, !dbg !1201
  %add153 = fadd float %v.9, %44, !dbg !1202
  br label %if.end154, !dbg !1203

if.end154:                                        ; preds = %if.then142.if.end154_crit_edge, %if.then146
  %.pre-phi10 = phi i32 [ %.pre9, %if.then142.if.end154_crit_edge ], [ %42, %if.then146 ], !dbg !1194
  %v.10 = phi float [ %v.9, %if.then142.if.end154_crit_edge ], [ %add153, %if.then146 ], !dbg !1112
  %and156 = and i32 %0, -7968, !dbg !1204
  %or1.i95 = or i32 %and156, 7951, !dbg !1205
  %45 = tail call i32 @llvm.hivm.shfl.idx.i32(i32 %.pre-phi10, i32 %or1.i95), !dbg !1205
  br label %if.end166, !dbg !1207

if.end166:                                        ; preds = %if.end154, %if.end140
  %v.11 = phi float [ %v.9, %if.end140 ], [ %v.10, %if.end154 ], !dbg !1112
  br i1 %cmp, label %if.then168, label %if.end171, !dbg !1208

if.then168:                                       ; preds = %if.end166
  %idxprom169 = sext i32 %add to i64, !dbg !1209
  %arrayidx170 = getelementptr inbounds float, ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)), i64 %idxprom169, !dbg !1209
  store float %v.11, ptr addrspace(6) %arrayidx170, align 4, !dbg !1210, !tbaa !940
  br label %if.end171, !dbg !1209

if.end171:                                        ; preds = %if.then168, %if.end166
  ret void, !dbg !1211
}
