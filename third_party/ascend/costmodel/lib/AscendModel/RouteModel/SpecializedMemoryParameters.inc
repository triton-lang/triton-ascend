// clang-format off
// Generated from the frozen 950PR profile. Do not edit coefficients.
constexpr const char *FrozenModelSHA256 = "c079ef9fccbbe623193a1ee26871816fe9f744c57d077dc39d5b6c748ac0213b";
constexpr unsigned FrozenParameterCounts[] = {60,60,60,60};
constexpr double DirectFragmentStartup = 0.3677911001734493;
static double predict_group0(const PublicFeatures &f) {
  const std::string head=smallRoute(f);
  double p=0.;
  const auto v=smallVector(f,head,64.0,true);
  if (head=="paged_masked") p=linear(v, {{0,0.7408846136205806}, {1,0.006568467152507573}, {2,-1.098342637365483e-05},
      {3,0.002352941417356855}, {6,0.026517644900473033}});
  else if (head=="paged_unmasked") p=linear(v, {{0,0.6737615443638392}, {1,0.00018677293686639756}, {2,5.143483479817672e-06},
      {3,0.013529238746279742}, {4,0.03390557134719121}});
  else if (head=="paged_zero") p=linear(v, {{0,0.7749375000000003}});
  else if (head=="segments_host") p=linear(v, {{0,1.0938643039795326}});
  else if (head=="segments_masked") p=linear(v, {{0,0.5985675600190099}, {1,0.04454388028055104}, {3,-0.00015508684334083507},
      {5,0.10544077463487184}, {7,0.011154304334043632}, {8,0.014463865944326137}, {9,-0.03063956302986622}, {10,-0.09329861375220695},
      {11,-3.579176423780413e-05}, {12,-0.00013106050295123208}});
  else if (head=="segments_other") p=linear(v, {{0,0.6538265191106886}, {1,0.002158498376414509}, {2,0.0022616003866150693},
      {3,0.020255442343706055}, {4,-0.00041066421269073686}, {5,-0.04169057207953746}, {6,0.03335147103906638}});
  else if (head=="segments_unit_few") p=linear(v, {{0,0.7714890482919432}, {1,-1.5129615697331214e-06}, {2,6.7173556494870115e-06},
      {3,0.00018066119081263784}, {4,3.7638693840668347e-06}, {5,-0.008941147327926}});
  else if (head=="segments_unit_many") p=linear(v, {{0,0.8845083066284989}, {2,0.0003587653836098273}, {3,0.00317516416466703},
      {4,-0.0012077927742565303}});
  else if (head=="segments_unknown") p=linear(v, {});
  else if (head=="segments_zero") p=linear(v, {{0,0.7692500000000002}});
  p += -8.251190185546917e-06 * (f.g.family==Family::Paged && f.g.storageBytes==1 && head=="paged_unmasked" && (!f.unitStep ||
      f.base[1]/f.base[3]<64.0) ? f.base[1] : 0.);
  p += adjustments(f,head, {{"segments_other","segments/proved_discrete_pitch_work",1.1785152324554307e-05},
      {"segments_other","segments/proved_regular_span",1.0226343369114517e-05},
      {"segments_unit_many","segments/proved_discrete_pitch_work",0.002478265527991377},
      {"segments_unit_many","segments/proved_regular_span",-1.7617354361416015e-05}, {"*","masked/logical_output_rows",0.03389062499999994},
      {"*","masked/host_logical_output_rows",0.009929687500000041}, {"*","masked/hidden_logical_output_rows",0.08884374999999999},
      {"*","segments/public_index_entries",0.023282587002558433}, {"*","segments/address_element_discrete_adds",5.893742735474627e-05},
      {"*","segments/address_element_regular_adds",3.451663777705015e-06}});
  if (std::isfinite(p)) p=std::max(0.71625,p);
  return p;
}
static double predict_group1(const PublicFeatures &f) {
  const std::string head=smallRoute(f);
  double p=0.;
  const auto v=smallVector(f,head,32.0,false);
  if (head=="paged_masked") p=linear(v, {{0,0.9177398754619908}, {1,0.004133339307898985}, {2,0.02448771180091583},
      {3,-0.034554312892442346}, {6,0.029235917434983882}});
  else if (head=="paged_unmasked") p=linear(v, {{0,0.19676583636836992}, {1,-0.018745960120479983}, {2,0.0003039618104107062},
      {3,0.7193269055671162}, {4,0.39766174557752354}});
  else if (head=="paged_zero") p=linear(v, {{0,1.0368125}});
  else if (head=="segments_host") p=linear(v, {{0,1.0389710001647718}});
  else if (head=="segments_masked") p=linear(v, {{0,0.7682171977014773}, {1,0.002905534176159526}, {3,0.024356518714085593},
      {5,0.0018656045970451424}, {7,-0.027569291338582658}, {8,-0.15294429133858287}, {9,0.10324107142857249}, {10,0.9990925729113935},
      {11,-0.00018450055803571764}, {12,0.009571374071258895}});
  else if (head=="segments_other") p=linear(v, {{0,-0.12412694136026647}, {1,0.09447789000428347}, {2,-0.012087383119026714},
      {3,-0.13060155348563937}, {4,-0.0008430960959295344}, {5,0.044242396471414945}, {6,-0.13608715884866704}});
  else if (head=="segments_unit_few") p=linear(v, {{0,1.005161150244412}, {1,-3.713983823857457e-05}, {2,0.0001463917724038837},
      {3,0.002220365106827031}, {4,-0.000200672839601527}, {5,-0.061446827212858934}});
  else if (head=="segments_unit_many") p=linear(v, {{0,0.9683372988566064}, {2,0.0006550738592770495}, {3,-0.00328497013979741},
      {4,-0.0017630061621631086}});
  else if (head=="segments_unknown") p=linear(v, {{0,7.055567000329544}});
  else if (head=="segments_zero") p=linear(v, {{0,0.7215625}});
  p += 0.09208663076664729 * (f.g.family==Family::Paged && f.g.storageBytes==1 && head=="paged_unmasked" && (!f.unitStep ||
      f.base[1]/f.base[3]<32.0) ? f.base[1] : 0.);
  p += adjustments(f,head, {{"segments_other","segments/proved_discrete_pitch_work",0.0007467513020968207},
      {"segments_other","segments/proved_regular_span",7.225801115800381e-05},
      {"segments_unit_many","segments/proved_discrete_pitch_work",0.005862242991904909},
      {"segments_unit_many","segments/proved_regular_span",-2.560679726651581e-05},
      {"segments_other","segments/proved_narrow_unit_runs",0.38202175416909523},
      {"segments_other","segments/proved_nonunit_extra_regular_axis_work",0.0001995098250854613},
      {"segments_masked","segments/proved_whole_row_active_segments",-0.029329822834645775},
      {"*","segments/public_index_entries",0.05700724995880704}, {"*","segments/address_element_discrete_adds",-0.00017205310777023685},
      {"*","segments/address_element_regular_adds",6.984839547032065e-06}});
  if (std::isfinite(p)) p=std::max(0.557125,p);
  return p;
}
static double predict_group2(const PublicFeatures &f) {
  const std::string head=(f.g.family==Family::Paged ? "paged" : (f.g.mask==Mask::None ? "segments_unmasked" : "segments_masked"));
  double p=0.;
  if (head=="paged") p=linear(f.page13, {{0,0.7922802592396254}, {1,0.0013422541600255672}, {2,0.0006378040032606104},
      {3,0.009545702388112847}, {4,0.0003899307508436422}, {5,-0.001065517438542088}, {6,0.05799031199230634}, {7,0.08541407137806523},
      {8,0.0008153381694455666}, {9,-0.0026798350337130165}, {11,0.00020706708497653532}, {12,0.06636772923751277}});
  else if (head=="segments_unmasked") p=linear(f.rich, {{0,0.7598966365951919}, {1,0.0013410006590399506}, {2,0.00018509462886845304},
      {3,0.0032586768835812054}, {4,0.00011060175866895408}, {5,-9.804569965892326e-05}, {6,0.007804220792333039},
      {11,-0.07165978823727383}, {12,0.16056430858318868}, {13,1.7164616006030316e-06}, {14,0.025891871033408072},
      {15,-0.01796731193165118}, {16,-0.04980301890519183}, {17,-0.008475053395249683}, {19,0.08473942883153984},
      {20,-0.014739737835126809}, {21,-0.0166179589639278}, {22,-0.029618426444549754}, {24,0.02237500267326661},
      {28,0.00014900456081591313}, {29,0.001254217595513554}});
  else if (head=="segments_masked") p=linear(f.simtMasked, {{0,1.0073026497631263}, {1,0.001235102278494353}, {2,0.000860783204055764},
      {3,0.0033271941489361467}, {4,-0.0002504321936262479}, {7,0.029554117434754664}, {8,0.017582834298115472}, {10,0.04702153764752137},
      {11,-0.002040962352479514}, {12,-0.2487065602836881}, {13,-0.3282547725170298}});
  p += adjustments(f,head, {{"paged","E_log2W",-0.00146344052474856}, {"paged","E_log2W_hinge2",0.0013106125319906278},
      {"segments/*","E_log2W",-0.0008046772134451455}, {"segments/*","E_log2W_hinge2",0.0007484877333679566},
      {"*","segments/public_index_entries",0.0017049948398051489}, {"*","segments/address_element_discrete_adds",-2.8657165056899935e-06},
      {"*","segments/address_element_regular_adds",-9.559250056952203e-05}});
  if (std::isfinite(p)) p=std::max(0.623375,p);
  return p;
}
static double predict_group3_fallback_segment_prior(const PublicFeatures &f) {
  double p=0.;
  p=linear(f.base, {{0,1.1647526763828044}, {1,0.0007999624554861009}, {3,0.002793820669201903}, {4,-0.0006347276677326516},
      {5,-0.00018401027016067302}, {6,-0.029381051637980746}, {7,-0.45562132327675003}});
  return p;
}
static double predict_group3_fallback(const PublicFeatures &f) {
  double p=0.;
  if (f.g.family==Family::Segments) p=predict_group3_fallback_segment_prior(f);
  else if (f.g.mask!=Mask::None) p=linear(f.page13, {{0,0.8677321428571428}, {8,0.0008410993303571423}, {11,1.6301836286272835e-05}});
  else if (f.g.mask==Mask::None) p=linear(f.page13, {{0,1.0594875}, {1,0.0020857898934431903}, {6,0.09310653543307085}});
  return p;
}
static double predict_group3(const PublicFeatures &f) {
  const std::string head=(wideRoute(f).empty() ? (f.g.family==Family::Segments ? "fallback/segments/shared" : (f.g.mask==Mask::None ?
      "fallback/paged/unmasked" : "fallback/paged/masked")) : "wide/"+wideRoute(f));
  double p=0.;
  const std::string route=wideRoute(f);
  if (route.empty()) p=predict_group3_fallback(f);
  else if (route=="few_discrete_axes") p=0.9401327593593496 + 0.012423962815204597 * f.base[6];
  else if (route=="several_discrete_axes") p=0.90932505000083;
  p += adjustments(f,head, {{"wide/few_discrete_axes","E_log2W",-0.00026897512290944105},
      {"wide/few_discrete_axes","E_log2W_hinge2",-1.2227293555055444e-05},
      {"wide/few_discrete_axes","E_log2W_hinge4",0.0001875930505428159}, {"wide/several_discrete_axes","E_log2W",-0.0003404492747944801},
      {"wide/several_discrete_axes","E_log2W_hinge2",0.00032961553456010485},
      {"wide/several_discrete_axes","E_log2W_hinge4",4.074096679687546e-05},
      {"fallback/segments/shared","E_log2W_hinge2",0.00032287191731980895},
      {"fallback/segments/shared","E_log2W_hinge4",0.00011610483852787411}, {"fallback/paged/unmasked","E_log2W",-0.001203442661555733},
      {"fallback/paged/unmasked","E_log2W_hinge2",0.0008601925959549546},
      {"fallback/paged/unmasked","E_log2W_hinge4",0.00026327337670513964}, {"wide/few_discrete_axes","E",0.0013893011703109128},
      {"wide/several_discrete_axes","E",0.0011880441008117607}, {"wide/few_discrete_axes","regular_segments",0.0009787946428571753},
      {"wide/several_discrete_axes","regular_segments",0.0009215960822224149},
      {"fallback/segments/shared","segments/proved_narrow_unit_runs",-0.00027110788733583543},
      {"fallback/segments/shared","segments/known_mask_payload_bytes",0.0009435003062365217},
      {"fallback/segments/shared","segments/known_mask_fraction",0.11954113132382788},
      {"fallback/segments/shared","segments/proved_whole_row_active_segments",-0.0003062871757517597},
      {"fallback/paged/masked","regular_segments",0.0012109375000000009},
      {"fallback/paged/unmasked","regular_segments",0.020286736704908957},
      {"fallback/segments/shared","segments/mask_prefix_indicator",-0.009550074351502291},
      {"fallback/segments/shared","segments/mask_rectangular_indicator",-0.0007434302930385828},
      {"*","segments/public_index_entries",0.0029078045259582105}, {"*","segments/address_element_discrete_adds",0.0002754211519605831},
      {"*","segments/address_element_regular_adds",6.072865699824382e-05}});
  if (std::isfinite(p)) p=std::max(0.591,p);
  return p;
}
static double predictFrozenGroup(const PublicFeatures &f,unsigned group) {
  switch (group) {
  case 0: return predict_group0(f);
  case 1: return predict_group1(f);
  case 2: return predict_group2(f);
  case 3: return predict_group3(f);
  default: return std::numeric_limits<double>::quiet_NaN();
  }
}
// clang-format on
