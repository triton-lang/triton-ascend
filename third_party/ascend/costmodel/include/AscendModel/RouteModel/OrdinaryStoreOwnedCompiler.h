// Finite compiler-produced conditional certificate for the owned SD API.
// This proves actual TTIR structure, not a live allocation or hardware result.
#ifndef ORDINARY_STORE_OWNED_COMPILER_H
#define ORDINARY_STORE_OWNED_COMPILER_H
#include "OrdinaryStoreRound4MainContract.h"
#include "OrdinaryStoreRound4VisibilityAnalysis.h"
#include "OrdinaryStoreSelected.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>

namespace mlir::ascend::ordinary_store_owned {
using namespace ordinary_store_main;
inline constexpr llvm::StringLiteral carrierName =
    "ascend.costmodel.ordinary_store.main_route_capability_json";
inline constexpr llvm::StringLiteral certificateKey =
    "ordinary_store_owned_conditional";
inline std::string str(const llvm::json::Object &o, StringRef key) {
  auto v = o.getString(key);
  return v ? v->str() : std::string{};
}
inline std::string sha(StringRef input) {
  auto d = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(input.data()), input.size()));
  return llvm::toHex(llvm::ArrayRef<uint8_t>(d), true);
}
inline std::string fingerprint(Operation *operation) {
  Operation *copy = operation->clone();
  copy->walk([](Operation *op) {
    SmallVector<StringAttr> remove;
    for (NamedAttribute a : op->getAttrs())
      if (a.getName().getValue().starts_with(
              "ascend.costmodel.ordinary_store."))
        remove.push_back(a.getName());
    for (auto a : remove)
      op->removeAttr(a);
  });
  std::string value;
  llvm::raw_string_ostream os(value);
  copy->print(os, OpPrintingFlags().useLocalScope());
  os.flush();
  copy->destroy();
  return sha(value);
}
// Same compact/sorted UTF-8 JSON contract as the Python recipe producer. The
// finite schema contains ASCII strings, integers, booleans, arrays and objects.
inline std::string canonical(const llvm::json::Value &value) {
  if (const auto *o = value.getAsObject()) {
    std::vector<std::string> keys;
    for (const auto &item : *o)
      keys.push_back(item.first.str());
    std::sort(keys.begin(), keys.end());
    std::string out = "{";
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i)
        out += ',';
      std::string quoted;
      llvm::raw_string_ostream os(quoted);
      os << llvm::json::Value(keys[i]);
      os.flush();
      out += quoted + ':' + canonical(*o->get(keys[i]));
    }
    return out + '}';
  }
  if (const auto *a = value.getAsArray()) {
    std::string out = "[";
    for (size_t i = 0; i < a->size(); ++i) {
      if (i)
        out += ',';
      out += canonical((*a)[i]);
    }
    return out + ']';
  }
  std::string out;
  llvm::raw_string_ostream os(out);
  os << value;
  os.flush();
  return out;
}
inline llvm::json::Object cloneObject(const llvm::json::Object &o) {
  return llvm::json::Object(o);
}
inline bool integers(const llvm::json::Object &o, StringRef key,
                     SmallVectorImpl<int64_t> &result, size_t count) {
  auto *a = o.getArray(key);
  if (!a || a->size() != count)
    return false;
  for (const auto &v : *a) {
    auto x = v.getAsInteger();
    if (!x)
      return false;
    result.push_back(*x);
  }
  return true;
}
inline bool one(const llvm::json::Object &o, StringRef key) {
  SmallVector<int64_t> v;
  return integers(o, key, v, 1) && v[0] == 1;
}
inline SmallVector<int64_t> shapeOf(Type type) {
  if (auto t = dyn_cast<RankedTensorType>(type))
    return SmallVector<int64_t>(t.getShape());
  return {};
}
inline int64_t elements(ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (auto d : shape)
    if (d < 1 || __builtin_mul_overflow(n, d, &n) || n > 128)
      return 0;
  return n;
}
// Bounded symbolic evaluation. Substitutions are ONLY the verified D-index
// load and verified public/scalar-load coefficient sources. A permutation is a
// runtime condition; substituting the identity here proves the separable map
// for every element in that domain, not the contents of any device tensor.
class Offsets {
  llvm::DenseMap<Value, std::optional<Tensor>> memo;
  unsigned remaining = 1024;
  static std::optional<Tensor> reshape(const Tensor &source, Type target,
                                       StringRef operation) {
    Tensor out;
    out.shape = shapeOf(target);
    auto count = elements(out.shape);
    if (!count || source.values.empty())
      return std::nullopt;
    if (operation == "tt.expand_dims" || operation == "arith.extsi") {
      if (source.values.size() != size_t(count))
        return std::nullopt;
      out.values = source.values;
      return out;
    }
    if (source.shape.empty()) {
      out.values.assign(count, source.values[0]);
      return out;
    }
    if (source.shape.size() != out.shape.size())
      return std::nullopt;
    for (int64_t k = 0; k < count; ++k) {
      int64_t flat = k, from = 0, stride = 1;
      for (size_t i = out.shape.size(); i-- > 0;) {
        auto coordinate = flat % out.shape[i];
        flat /= out.shape[i];
        if (source.shape[i] != 1 && source.shape[i] != out.shape[i])
          return std::nullopt;
        from += (source.shape[i] == 1 ? 0 : coordinate) * stride;
        stride *= source.shape[i];
      }
      if (from < 0 || size_t(from) >= source.values.size())
        return std::nullopt;
      out.values.push_back(source.values[from]);
    }
    return out;
  }
  static bool fits(Type type, int64_t value) {
    if (auto t = dyn_cast<RankedTensorType>(type))
      type = t.getElementType();
    auto i = dyn_cast<IntegerType>(type);
    if (!i || i.getWidth() < 2 || i.getWidth() > 64)
      return false;
    return i.getWidth() == 64 ||
           (value >= -(int64_t(1) << (i.getWidth() - 1)) &&
            value <= (int64_t(1) << (i.getWidth() - 1)) - 1);
  }

public:
  void bind(Value value, Tensor tensor) { memo[value] = std::move(tensor); }
  std::optional<Tensor> integer(Value value, unsigned depth = 0) {
    if (auto i = memo.find(value); i != memo.end())
      return i->second;
    if (depth > 48 || !remaining)
      return std::nullopt;
    --remaining;
    auto *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    const auto name = op->getName().getStringRef();
    std::optional<Tensor> result;
    if (name == "arith.constant" || name == "tt.make_range") {
      StaticMask staticValues;
      result = staticValues.evaluate(value);
    } else if (op->getNumOperands() == 1 &&
               (name == "tt.splat" || name == "tt.broadcast" ||
                name == "tt.expand_dims" || name == "arith.extsi")) {
      auto source = integer(op->getOperand(0), depth + 1);
      if (source)
        result = reshape(*source, value.getType(), name);
    } else if (op->getNumOperands() == 2 &&
               (name == "arith.addi" || name == "arith.muli")) {
      auto a = integer(op->getOperand(0), depth + 1),
           b = integer(op->getOperand(1), depth + 1);
      const auto shape = shapeOf(value.getType());
      const auto count = elements(shape);
      if (a && b && count && a->shape == shape && b->shape == shape &&
          a->values.size() == size_t(count) &&
          b->values.size() == size_t(count)) {
        Tensor out;
        out.shape = shape;
        bool valid = true;
        for (int64_t k = 0; k < count; ++k) {
          int64_t z;
          if ((name == "arith.addi"
                   ? __builtin_add_overflow(a->values[k], b->values[k], &z)
                   : __builtin_mul_overflow(a->values[k], b->values[k], &z)) ||
              !fits(value.getType(), z)) {
            valid = false;
            break;
          }
          out.values.push_back(z);
        }
        if (valid)
          result = std::move(out);
      }
    }
    if (result && (result->values.size() > 128 ||
                   llvm::any_of(result->values, [&](int64_t x) {
                     return !fits(value.getType(), x);
                   })))
      result.reset();
    memo[value] = result;
    return result;
  }
  std::optional<Tensor> pointer(Value value, Value root, unsigned depth = 0) {
    if (depth > 48 || !remaining)
      return std::nullopt;
    --remaining;
    if (value == root)
      return Tensor{{}, {0}};
    auto *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    auto name = op->getName().getStringRef();
    if (op->getNumOperands() == 1 &&
        (name == "tt.splat" || name == "tt.broadcast" ||
         name == "tt.expand_dims")) {
      auto src = pointer(op->getOperand(0), root, depth + 1);
      return src ? reshape(*src, value.getType(), name) : std::nullopt;
    }
    if (name != "tt.addptr" || op->getNumOperands() != 2)
      return std::nullopt;
    auto a = pointer(op->getOperand(0), root, depth + 1),
         b = integer(op->getOperand(1));
    if (!a || !b || a->shape != b->shape ||
        a->values.size() != b->values.size())
      return std::nullopt;
    for (size_t k = 0; k < a->values.size(); ++k) {
      int64_t z;
      if (__builtin_add_overflow(a->values[k], b->values[k], &z) || z < 0 ||
          z > INT32_MAX)
        return std::nullopt;
      a->values[k] = z;
    }
    return a;
  }
};
struct Proof {
  bool eligible = false;
  std::string reason = "owned_recipe_missing";
  llvm::json::Object certificate;
};
inline Proof prove(ModuleOp module, const llvm::json::Object &cap,
                   StringRef profileSha, StringRef unitSha) {
  Proof out;
  auto fail = [&](const char *r) {
    out.reason = r;
    return std::move(out);
  };
  auto *recipe = cap.getObject("ordinary_store_owned_recipe");
  if (!recipe)
    return out;
  // The issuer defines a 1:1 SYS_CNT/resource-system-cycle bridge ONLY for
  // these actual already-audited loader identities. A future profile/unit
  // definition must establish its own bridge; its familiar target name or a
  // syntactically valid SHA is insufficient. The old native API is unchanged.
  if (profileSha !=
          "a4e38540af7ec47d5085fd2006734fe6eaaf5a15246f48280555e1cdca96b372" ||
      unitSha !=
          "dd21bc551513ff0a03fa055aa3d6770f5e8c0a48a92ba9def62ca17fe48489c4")
    return fail("owned_unit_issuer_profile_outside_audited_SYS_CNT_pair");
  const auto visibility = str(*recipe, "visibility"),
             route = str(*recipe, "fixed_route");
  if (str(*recipe, "schema") != "owned_sd_compile_recipe_v1" ||
      str(*recipe, "runtime_guard_abi") !=
          "owned-sd-precompiled-eager-launch-v1" ||
      (visibility != "known" && visibility != "host" &&
       visibility != "device") ||
      (route != "all_simd" && route != "all_simt_only") ||
      str(cap, "backend_forced_route_candidate") != route ||
      !indirect_store_round4::main_ta::sha256(str(*recipe, "source_sha256")) ||
      !indirect_store_round4::main_ta::sha256(
          str(*recipe, "source_key_sha256")) ||
      !indirect_store_round4::main_ta::sha256(profileSha.str()) ||
      !indirect_store_round4::main_ta::sha256(unitSha.str()))
    return fail("owned_recipe_schema_source_or_fixed_route_invalid");
  auto normalized = cloneObject(*recipe);
  normalized.erase("source_sha256");
  normalized.erase("recipe_sha256");
  if (sha(canonical(llvm::json::Value(std::move(normalized)))) !=
      str(*recipe, "recipe_sha256"))
    return fail("owned_recipe_digest_mismatch");
  if (cap.getBoolean("layout_merge_applied") != false ||
      cap.getInteger("layout_coalescing_factor") != int64_t(1) ||
      cap.getInteger("layout_coalescing_axis") != int64_t(-1) ||
      !one(cap, "whole_kernel_superblock_factors") ||
      !one(cap, "scope_superblock_factors") ||
      cap.getInteger("source_logical_program_count_hint") != int64_t(1))
    return fail("owned_transform_or_single_program_contract_missing");
  SmallVector<int64_t> shape, lower, upper, coefficients, mapping, hostArgs;
  if (!integers(*recipe, "shape", shape, 2) ||
      !integers(*recipe, "lower", lower, 2) ||
      !integers(*recipe, "upper", upper, 2) ||
      !integers(*recipe, "coefficients", coefficients, 2) ||
      !recipe->getBoolean("has_mask"))
    return fail("owned_geometry_recipe_invalid");
  const int64_t S = shape[0], D = shape[1], PS = coefficients[0],
                PD = coefficients[1];
  if (S < 2 || D < 2 || (S & (S - 1)) || (D & (D - 1)) || S > 64 || D > 64 ||
      S * D > 128 || PS < 1 || PD < 1 || PS > INT32_MAX || PD > INT32_MAX ||
      lower[0] < 0 || lower[1] < 0 || lower[0] > upper[0] ||
      lower[1] > upper[1] || upper[0] > S || upper[1] > D ||
      recipe->getInteger("element_bytes") != int64_t(4) ||
      (str(*recipe, "dtype") != "float32" && str(*recipe, "dtype") != "int32"))
    return fail("owned_geometry_or_dtype_outside_finite_domain");
  int64_t a, b, payload;
  if (__builtin_mul_overflow(S - 1, PS, &a) ||
      __builtin_mul_overflow(D - 1, PD, &b) ||
      __builtin_add_overflow(a, b, &payload) ||
      __builtin_add_overflow(payload, int64_t(1), &payload) ||
      payload > INT32_MAX || recipe->getInteger("payload_elements") != payload)
    return fail("owned_payload_or_signed_i32_bounds_invalid");
  std::set<int64_t> unique;
  for (int64_t s = 0; s < S; ++s)
    for (int64_t d = 0; d < D; ++d)
      unique.insert(s * PS + d * PD);
  if (unique.size() != size_t(S * D))
    return fail("owned_target_offsets_alias");
  auto *domain = recipe->getObject("index_domain");
  if (!domain || domain->getInteger("minimum") != int64_t(0) ||
      domain->getInteger("maximum") != D - 1 ||
      domain->getBoolean("permutation") != true ||
      domain->getInteger("length") != D)
    return fail("owned_permutation_requirement_missing");
  const size_t arity = visibility == "known"    ? 3
                       : visibility == "device" ? 4
                                                : 5;
  if (!integers(*recipe, "compact_to_full", mapping, arity) ||
      !integers(*recipe, "host_coefficient_arguments", hostArgs,
                visibility == "host" ? 2 : 0) ||
      recipe->getInteger("target_argument") != int64_t(0) ||
      recipe->getInteger("index_argument") != int64_t(1) ||
      recipe->getInteger("value_argument") != int64_t(2) ||
      recipe->getInteger("coefficient_argument") !=
          (visibility == "device" ? int64_t(3) : int64_t(-1)))
    return fail("owned_compact_argument_mapping_invalid");
  for (size_t i = 0; i < mapping.size(); ++i)
    if (mapping[i] < 0 || (i && mapping[i] <= mapping[i - 1]))
      return fail("owned_full_positional_mapping_invalid");
  if (visibility == "host" && hostArgs != SmallVector<int64_t>({3, 4}))
    return fail("owned_host_arguments_invalid");
  triton::FuncOp entry;
  unsigned entries = 0;
  bool allowed = true;
  module.walk([&](triton::FuncOp f) {
    entry = f;
    ++entries;
  });
  if (entries != 1 || !entry || entry.getBody().empty() ||
      !llvm::hasSingleElement(entry.getBody()) ||
      entry.getNumArguments() != arity ||
      (entry->getAttrOfType<StringAttr>("sym_visibility") &&
       entry->getAttrOfType<StringAttr>("sym_visibility").getValue() !=
           "public"))
    return fail("owned_unique_public_entry_invalid");
  SmallVector<triton::StoreOp> stores;
  SmallVector<triton::LoadOp> loads;
  entry.walk([&](Operation *op) {
    auto n = op->getName().getStringRef();
    if (op != entry.getOperation() && op->getNumRegions())
      allowed = false;
    if (!llvm::is_contained({StringRef("tt.func"), StringRef("tt.return"),
                             StringRef("arith.constant"),
                             StringRef("tt.make_range"), StringRef("tt.splat"),
                             StringRef("tt.broadcast"),
                             StringRef("tt.expand_dims"),
                             StringRef("arith.extsi"), StringRef("arith.addi"),
                             StringRef("arith.muli"), StringRef("tt.addptr"),
                             StringRef("arith.cmpi"), StringRef("arith.andi"),
                             StringRef("tt.load"), StringRef("tt.store")},
                            n))
      allowed = false;
    if (auto s = dyn_cast<triton::StoreOp>(op))
      stores.push_back(s);
    if (auto l = dyn_cast<triton::LoadOp>(op))
      loads.push_back(l);
  });
  if (!allowed || stores.size() != 1 ||
      loads.size() != (visibility == "device" ? 4 : 2))
    return fail("owned_extra_effect_or_unsupported_expression");
  auto store = stores[0];
  auto tensor = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!tensor || SmallVector<int64_t>(tensor.getShape()) != shape ||
      (str(*recipe, "dtype") == "float32"
           ? !tensor.getElementType().isF32()
           : !tensor.getElementType().isInteger(32)))
    return fail("owned_actual_store_shape_or_type_mismatch");
  if (store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL ||
      !store.getBoundaryCheck().empty() ||
      (store->getAttrOfType<BoolAttr>("isVolatile") &&
       store->getAttrOfType<BoolAttr>("isVolatile").getValue()))
    return fail("owned_target_store_policy_not_supported");
  auto pointerType = [&](unsigned i, Type element) {
    auto t = dyn_cast<triton::PointerType>(entry.getArgument(i).getType());
    return t && t.getPointeeType() == element;
  };
  if (!pointerType(0, tensor.getElementType()) ||
      !pointerType(1, IntegerType::get(module.getContext(), 32)) ||
      !pointerType(2, tensor.getElementType()) ||
      (visibility == "device" &&
       !pointerType(3, IntegerType::get(module.getContext(), 32))))
    return fail("owned_actual_pointer_types_mismatch");
  auto terms = visibilityAxes(store.getPtr(), entry);
  if (terms.size() != 2 || !terms[0].valid || !terms[1].valid ||
      !terms[0].coordinate || !terms[1].coordinate || terms[0].loadedIndex ||
      !terms[1].loadedIndex)
    return fail("owned_actual_geometry_is_not_SD");
  auto index =
      dyn_cast_or_null<triton::LoadOp>(terms[1].loadedIndex.getDefiningOp());
  auto values =
      dyn_cast_or_null<triton::LoadOp>(store.getValue().getDefiningOp());
  auto indexType =
      index ? dyn_cast<RankedTensorType>(index.getType()) : RankedTensorType{};
  if (!index || !values || !indexType || indexType.getRank() != 1 ||
      indexType.getDimSize(0) != D || !indexType.getElementType().isInteger(32))
    return fail("owned_exact_index_or_value_load_missing");
  for (auto load : loads) {
    if (load.getMask() || !load.getBoundaryCheck().empty() ||
        load.getCache() != triton::CacheModifier::NONE ||
        load.getEvict() != triton::EvictionPolicy::NORMAL ||
        (load->getAttrOfType<BoolAttr>("isVolatile") &&
         load->getAttrOfType<BoolAttr>("isVolatile").getValue()))
      return fail("owned_load_policy_not_supported");
  }
  Offsets offsets;
  for (unsigned i = 0; i < 2; ++i) {
    const auto &pitch = terms[i].pitch;
    const auto expected = coefficients[i];
    if (visibility == "known") {
      if (pitch.kind != PitchKind::Constant || pitch.scale != expected)
        return fail("owned_known_pitch_IR_mismatch");
    } else if (visibility == "host") {
      if (pitch.kind != PitchKind::HostArgument || pitch.argument != 3 + i ||
          pitch.scale != 1 || hostPitchValue(pitch, entry, cap) != expected)
        return fail("owned_host_actual_scalar_binding_mismatch");
      offsets.bind(pitch.source, Tensor{{}, {expected}});
    } else {
      if (pitch.kind != PitchKind::DeviceScalarLoad || pitch.scale != 1)
        return fail("owned_device_scalar_source_missing");
      auto coefficient =
          dyn_cast_or_null<triton::LoadOp>(pitch.source.getDefiningOp());
      auto position = coefficient ? offsets.pointer(coefficient.getPtr(),
                                                    entry.getArgument(3))
                                  : std::nullopt;
      if (!coefficient || !coefficient.getType().isInteger(32) || !position ||
          !position->shape.empty() ||
          position->values != SmallVector<int64_t>({int64_t(i)}))
        return fail("owned_device_coefficient_exact_arg_offset_mismatch");
      offsets.bind(pitch.source, Tensor{{}, {expected}});
    }
  }
  auto ioff = offsets.pointer(index.getPtr(), entry.getArgument(1));
  auto voff = offsets.pointer(values.getPtr(), entry.getArgument(2));
  if (!ioff || ioff->shape != SmallVector<int64_t>({D}) || !voff ||
      voff->shape != shape)
    return fail("owned_input_pointer_bounds_unproved");
  for (int64_t i = 0; i < D; ++i)
    if (ioff->values[i] != i)
      return fail("owned_index_load_not_exact_domain");
  for (int64_t i = 0; i < S * D; ++i)
    if (voff->values[i] != i)
      return fail("owned_value_load_not_exact_row_major");
  Tensor identity;
  identity.shape = {D};
  for (int64_t d = 0; d < D; ++d)
    identity.values.push_back(d);
  offsets.bind(index.getResult(), std::move(identity));
  auto target = offsets.pointer(store.getPtr(), entry.getArgument(0));
  if (!target || target->shape != shape ||
      target->values.size() != size_t(S * D))
    return fail("owned_target_pointer_unproved");
  for (int64_t s = 0; s < S; ++s)
    for (int64_t d = 0; d < D; ++d)
      if (target->values[s * D + d] != s * PS + d * PD)
        return fail("owned_target_contains_unbound_offset_or_geometry");
  // Every load must be one of the exact three input domains proved above.
  for (auto load : loads)
    if (load != index && load != values &&
        load.getResult() != terms[0].pitch.source &&
        load.getResult() != terms[1].pitch.source)
      return fail("owned_unaccounted_input_load");
  const bool hasMask = *recipe->getBoolean("has_mask");
  if (bool(store.getMask()) != hasMask)
    return fail("owned_source_mask_presence_changed");
  if (!hasMask && (lower != SmallVector<int64_t>({0, 0}) || upper != shape))
    return fail("owned_unmasked_bounds_mismatch");
  if (hasMask) {
    StaticMask evaluator;
    auto mask = evaluator.evaluate(store.getMask());
    if (!mask || mask->shape != shape || mask->values.size() != size_t(S * D))
      return fail("owned_mask_not_static_rectangle");
    for (int64_t s = 0; s < S; ++s)
      for (int64_t d = 0; d < D; ++d)
        if (bool(mask->values[s * D + d]) !=
            (lower[0] <= s && s < upper[0] && lower[1] <= d && d < upper[1]))
          return fail("owned_actual_mask_bounds_mismatch");
  }
  out.certificate = llvm::json::Object{
      {"schema", "ordinary_store_owned_compiler_certificate_v1"},
      {"enabled", true},
      {"invocation_scope", "conditional_owned_eager_first_store_v1"},
      {"selected_SIMD_model_sha256",
       indirect_store_selected::selectedSimdModelSha256},
      {"selected_SIMT_model_sha256",
       indirect_store_selected::selectedSimtModelSha256},
      {"selected_freeze_sha256", indirect_store_selected::selectedFreezeSha256},
      {"ir_contract", "mlir-local-no-debug-ordinary-attrs-stripped-v1"},
      {"module_fingerprint_sha256", fingerprint(module)},
      {"store_fingerprint_sha256", fingerprint(store)},
      {"entry_pointer_argument", int64_t(0)},
      {"store_ordinal", int64_t(0)},
      {"recipe_sha256", str(*recipe, "recipe_sha256")},
      {"source_sha256", str(*recipe, "source_sha256")},
      {"source_key_sha256", str(*recipe, "source_key_sha256")},
      {"visibility", visibility},
      {"fixed_route", route},
      {"mask_representation", hasMask ? "rectangular" : "unmasked"},
      {"profile_sha256", profileSha.str()},
      {"unit_source_sha256", unitSha.str()},
      {"unit_contract", "syscnt_ticks_to_active_profile_system_cycles_v1"},
      {"syscnt_ticks_per_system_cycle", int64_t(1)},
      {"increment_contract", "round4_first_store_matched_increment_syscnt_v1"},
      {"accounting_boundary", "round4_matched_target_only"},
      {"runtime_guard_abi", "owned-sd-precompiled-eager-launch-v1"},
      {"required_grid", llvm::json::Array{1, 1, 1}},
      {"runtime_guard_required", true},
      {"requires_owned_uploaded_permutation_and_coefficients", true},
      {"requires_per_launch_fresh_guards_only_output", true},
      {"requires_completion_retention", true},
      {"allocation_observed", false},
      {"hardware_acceptance_claimed", false}};
  out.eligible = true;
  out.reason = "actual_TTIR_owned_SD_conditional_certificate";
  return out;
}
inline bool verify(ModuleOp module, const llvm::json::Object &cap,
                   const llvm::json::Object &certificate, StringRef profileSha,
                   StringRef unitSha) {
  auto actual = prove(module, cap, profileSha, unitSha);
  return actual.eligible &&
         canonical(llvm::json::Value(std::move(actual.certificate))) ==
             canonical(llvm::json::Value(cloneObject(certificate)));
}
} // namespace mlir::ascend::ordinary_store_owned
#endif
