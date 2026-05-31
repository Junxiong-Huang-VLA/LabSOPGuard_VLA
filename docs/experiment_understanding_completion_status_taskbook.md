# 瀹為獙鐞嗚В鑳藉姏瀹屾垚鐘舵€佷笌浠诲姟涔?

## 鍩哄噯浜х墿

- 鍩哄噯杩愯锛歚.runtime/timeline_demo`
- 涓绘ā鍨嬶細`yolo26s_pose_lab_v4_focus_auto`
- 宸插彂鐜版ā鍨嬩骇鐗╋細42 涓?
- 宸插彂鐜版暟鎹泦閰嶇疆锛? 涓?
- 涓?YOLO 鏁版嵁闆嗭細4634 寮犲浘鐗囥€?634 涓爣绛俱€?3 绫?
- 缁熶竴鏃堕棿绾夸簨浠讹細36 鏉?
- 楂樼骇瑙嗚璇佹嵁锛? 鏉?
- 缁撴瀯鍖栬棰戠悊瑙ｄ簨浠讹細26 鏉?
- 瀹為獙姝ラ锛? 涓紝宸插畬鎴?3 涓紝鏈瀵?1 涓?
- 浜哄伐纭闃熷垪锛? 鏉?pending

## 涓€銆佽棰戠墿鐞嗚瘉鎹笌浜嬩欢璇嗗埆

| 鑳藉姏 | 褰撳墠瀹屾垚鎯呭喌 | 瀹屾垚鏁堟灉 | 鏈畬鎴?/ 寰呭姞寮?|
| --- | --- | --- | --- |
| 鎵嬮儴涓庣墿浣撴帴瑙︽娴?| 宸插畬鎴愬悗绔兘鍔涳紱YOLO 妫€娴?+ hand-object interaction + micro-segment 缁戝畾 | demo 杈撳嚭 `hand_object_contact=3`锛岀湡瀹?session 鏇捐緭鍑?83 涓墜鐗╀氦浜?| 澶嶆潅閬尅銆佸悓绫诲鐗╀綋韬唤淇濇寔浠嶉渶鏇村己璺熻釜鍣ㄥ拰璇勬祴闆?|
| 鐗╀綋绉诲姩妫€娴?| 宸插寮猴紱浼樺厛浣跨敤 `track_id`锛屾棤 `track_id` 鏃跺仛 bbox 鏃跺簭鍏宠仈 | 鍙緭鍑?`object_trajectory_movement`锛沝emo 褰撳墠杈撳嚭 `object_movement_candidate=3` | demo 璇佹嵁浠嶅涓?candidate锛岀湡瀹炵Щ鍔ㄧ‘璁や緷璧栧抚绾?track/bbox 璐ㄩ噺 |
| 娑蹭綋杞Щ妫€娴?| 宸叉敮鎸佸姩浣?涓婁笅鏂囧€欓€夊拰娑蹭綅鍏抽敭甯у垎鏋?| demo 杈撳嚭 `liquid_transfer_candidate=1`銆乣liquid_flow_candidate_visual=1` | 褰撳墠妯″瀷娓呭崟鏈彂鐜?liquid/stream/meniscus 鍒嗗壊绫伙紝涓嶈兘寮虹‘璁ゆ恫娴?|
| 璁惧闈㈡澘鎿嶄綔妫€娴?| 宸叉敮鎸?balance/panel 绫诲€欓€夊拰鍙€?OCR | demo 杈撳嚭 `equipment_panel_operation_candidate=6` | 褰撳墠鏈彂鐜?button/knob/display 鐘舵€佺被锛涙寜閽?鏃嬮挳鐘舵€佽瘑鍒湭寮虹‘璁?|
| 瀹瑰櫒鐘舵€佸彉鍖栨娴?| 宸叉敮鎸佸鍣ㄤ氦浜掋€侀鑹插彉鍖栥€乧ap/lid token锛涘凡鎺?`tube-cap/tube_cap` | demo 杈撳嚭 `container_state_change_candidate=6`銆乣container_open_close=3` | 寮€/鍏崇洊鐘舵€佹湰韬粛闇€鏃跺簭鐘舵€佸垎绫绘垨寮€鐩?鍏崇洊鏍囨敞 |
| 瀹為獙鍔ㄤ綔鍒嗙被 | 宸插畬鎴愯鍒欏寲鍒嗙被锛屾潵鑷?micro-segment 鏂囨湰鎻忚堪銆佸璞°€佷氦浜掕瘉鎹?| demo 杈撳嚭 `experiment_action_classification=3` | 澶嶆潅鍔ㄤ綔杈圭晫銆佺浉浼煎姩浣滅粏鍒嗛渶瑕佽缁冨姩浣滃垎绫诲櫒鎴?VLM 鏍￠獙 |
| 鐗╀綋鐘舵€佸彉鍖栬褰?| 宸插畬鎴愮姸鎬佺储寮曞拰瑙嗛鐞嗚В浜嬩欢 | demo 杈撳嚭 `object_state_change=3`锛岀姸鎬佸彉鍖栧彲杩涘叆璇佹嵁閾?| 娑蹭綅銆侀鑹层€佸紑鐩栫姸鎬佷粛闇€鏇村己瑙嗚妯″瀷 |
| 浜嬩欢缃俊搴﹁绠?| 宸插畬鎴愶紱姣忔潯 video event 鍜?process step 閮芥湁 confidence | demo 姝ラ缃俊搴︼細1.0銆?.77銆?.62銆?.0 | 缃俊搴︿粛鏄鍒欒瀺鍚堬紝缂哄皯鍩轰簬楠岃瘉闆嗘牎鍑嗙殑姒傜巼鏍囧畾 |
| 寮傚父浜嬩欢鏍囪 | 宸插畬鎴愶紱閲嶅銆侀『搴忓啿绐併€佷綆缃俊搴︺€佹湭瑙傚療杩涘叆 flags/queue | demo 鏍囪 `unexpected_repeat`銆乣observed_out_of_sop_order`銆乣no_direct_video_observation` | 寮傚父绫诲瀷搴撻渶瑕佺户缁墿鍏咃紝渚嬪鍗遍櫓鎿嶄綔銆佽秴鏃躲€佸墏閲忓亸宸?|

## 浜屻€佺粨鏋勫寲瑙嗛鐞嗚В

| 鑳藉姏 | 褰撳墠瀹屾垚鎯呭喌 | 瀹屾垚鏁堟灉 | 鏈畬鎴?/ 寰呭姞寮?|
| --- | --- | --- | --- |
| 璇嗗埆瀹為獙瀵硅薄 | 宸插畬鎴愶紱鍩轰簬 YOLO 绫诲埆鍜?schema | 宸茶瘑鍒?balance銆乻ample_bottle銆乸ipette銆乼ube_cap 绛夎兘鍔?| 灏忕洰鏍囥€侀伄鎸″拰鍚岀被澶氱墿浣?ReID 闇€澧炲己 |
| 璇嗗埆瀹為獙鍔ㄤ綔 | 宸插畬鎴愶紱杈撳嚭 action_type | demo 璇嗗埆 weighing銆乸ipetting銆乻ample_handling | 澶嶆潅鍔ㄤ綔缁勫悎闇€瑕佸姩浣滄ā鍨?VLM 浜屾鍒ゅ埆 |
| 璇嗗埆瀹為獙鐘舵€?| 宸插畬鎴愬熀纭€鐘舵€侊紱鎺ヨЕ寮€濮嬨€佸嘲鍊笺€侀噴鏀俱€佺姸鎬佸彉鍖?| 缁熶竴鏃堕棿绾挎湁 micro contact/peak/release anchor | 娑蹭綋鐘舵€併€佽澶囩姸鎬併€佸鍣ㄥ紑闂姸鎬佽繕闇€鐘舵€佹ā鍨?|
| 鐢熸垚瑙嗛璇箟鎻忚堪 | 宸插畬鎴愶紱segment/micro-segment 鏈夋枃鏈弿杩板拰 index text | 鍙繘鍏ュ悜閲忔绱?| 鏂囨璐ㄩ噺渚濊禆瑙勫垯妯℃澘锛屽鏉傛弿杩伴渶 VLM 鐢熸垚涓庢牎楠?|
| 杈撳嚭缁撴瀯鍖栬棰戠悊瑙ｇ粨鏋?| 宸插畬鎴?| `metadata/video_understanding.jsonl`銆乻ummary JSON | JSON schema 鍙户缁増鏈寲鍜屽彂甯冩帴鍙ｅ绾?|
| 涓庡叧閿抚銆佽棰戠墖娈电粦瀹?| 宸插畬鎴?| asset catalog 96 涓礌鏉愶紝姝ラ璇佹嵁鍚?clip/keyframe refs | 褰撳墠鐪熷疄 session 鑻ユ湭杩愯瀹屾暣 pipeline锛岀礌鏉?catalog 鍙兘涓虹┖ |
| 涓哄悗缁楠ゆ帹鐞嗘彁渚涜緭鍏?| 宸插畬鎴?| process reasoner 宸叉秷璐?video_understanding/state/context | 闇€瑕佹洿澶?SOP 鍦烘櫙鍥炲綊闆?|

## 涓夈€佸妯℃€佷笂涓嬫枃铻嶅悎

| 鑳藉姏 | 褰撳墠瀹屾垚鎯呭喌 | 瀹屾垚鏁堟灉 | 鏈畬鎴?/ 寰呭姞寮?|
| --- | --- | --- | --- |
| 瑙ｆ瀽鐢ㄦ埛鏂囨湰 | 宸插畬鎴愶紱缁熶竴鏃堕棿绾挎敮鎸?user_text | demo 鏈緭鍏ラ澶?user_events锛屽洜姝ゆ姤鍛婁负 0 | 闇€瑕佹帴鐪熷疄浜や簰鏃ュ織 |
| 瑙ｆ瀽璇煶杞啓 | 宸插畬鎴?| demo transcript_rows=3 | 璇煶 ASR 缃俊搴﹀拰璇磋瘽浜轰俊鎭湭鍏呭垎鍒╃敤 |
| 瑙ｆ瀽 AI 鍘嗗彶鍥炲 | 宸插畬鎴愯緭鍏ョ粨鏋?| demo 鏈緭鍏?ai_events锛屽洜姝や负 0 | 闇€瑕佹帴鐪熷疄 AI 鍥炲鏃ュ織 |
| 瑙ｆ瀽鐢ㄦ埛涓婁紶鍥炬枃 | 宸插畬鎴愯緭鍏ョ粨鏋?| demo 鏈緭鍏?uploads锛屽洜姝や负 0 | 闇€瑕佹帴涓婁紶鏂囦欢鍏冩暟鎹拰 OCR/VLM 缁撴灉 |
| 妫€绱㈠凡鏈夋暟鎹簱 | 宸叉敮鎸佹湰鍦?JSON/JSONL database_paths | demo database_rows=0 | 鏈帴 SQL/LIMS/杩滅▼鏁版嵁搴?|
| 鐞嗚В瀹為獙鐩殑銆佹祦绋嬨€佹潗鏂欍€佸弬鏁?| 宸插畬鎴愬熀纭€鎶藉彇 | demo purpose=1銆乸rocedure=4銆乵aterials=5銆乸arameters=62 | 鍙傛暟鎶藉彇闇€鍗曚綅瑙勮寖鍖栧拰 SOP 鍙傛暟绾︽潫鏍￠獙 |
| 寤虹珛褰撳墠瀹為獙涓婁笅鏂?| 宸插畬鎴?| `metadata/experiment_context.json` | 闇€瑕佹洿澶氬閮ㄥ巻鍙插拰鏁版嵁搴撹ˉ寮?|
| 涓庤棰戠粨鏋滆瀺鍚?| 宸插畬鎴?| context source_counts 鍚?timeline/video/material/transcript | 鍐茬獊娑堣В浠嶄富瑕佽鍒欏寲 |

## 鍥涖€佹楠ょ姸鎬佹満涓庢帹鐞?

| 鑳藉姏 | 褰撳墠瀹屾垚鎯呭喌 | 瀹屾垚鏁堟灉 | 鏈畬鎴?/ 寰呭姞寮?|
| --- | --- | --- | --- |
| 寤虹珛瀹為獙姝ラ鐘舵€佹満 | 宸插畬鎴?| `metadata/experiment_process.json` | 澶嶆潅宓屽 SOP DSL 鏈畬鎴?|
| 瀹氫箟姝ラ杩涘叆鏉′欢 | 宸叉敮鎸?entry_conditions/branch_condition | 鍙鐞?branch not taken | 鏉′欢琛ㄨ揪寮忚兘鍔涙湁闄?|
| 瀹氫箟姝ラ瀹屾垚鏉′欢 | 宸叉敮鎸?completion_conditions | 姣忔杈撳嚭 completion_conditions | 灏氭湭鎺ュ己鍙傛暟闃堝€煎垽瀹?|
| 鍒ゆ柇褰撳墠姝ラ | 宸插畬鎴?| demo current=step_004 | 闇€瑕佸疄鏃舵祦寮忔洿鏂版椂鐨?debounce |
| 鍒ゆ柇宸插畬鎴愭楠?| 宸插畬鎴?| demo completed=3 | 浣庣疆淇″畬鎴愰渶浜哄伐纭 |
| 鎺ㄧ悊涓嬩竴姝?| 宸插畬鎴愬熀纭€ next_step | demo next=step_004 | 鍒嗘敮 SOP 鐨勬渶浼樹笅涓€姝ヤ粛闇€瑙勫垯寮曟搸澧炲己 |
| 澶勭悊璺虫銆侀噸澶嶃€佸紓甯搞€佸啿绐?| 宸插畬鎴愰儴鍒?| demo 鎹曡幏 unexpected_repeat銆乷ut_of_sop_order | 澶嶆潅鍐茬獊娑堣В绛栫暐闇€澧炲己 |
| 杈撳嚭姝ラ缃俊搴?| 宸插畬鎴?| 姣忎釜 step 鏈?confidence | 缂哄皯鍘嗗彶楠岃瘉闆嗘鐜囨牎鍑?|
| 浜哄伐纭鎴栬嚜鍔ㄧ‘璁?| 鍚庣宸插畬鎴?| queue 4 鏉?pending锛屽彲鍐欏洖 decision | 鍓嶇瀹℃壒 UI 鏈畬鎴?|

## 浜斻€佺己澶辨楠よˉ鍏?

| 鑳藉姏 | 褰撳墠瀹屾垚鎯呭喌 | 瀹屾垚鏁堟灉 | 鏈畬鎴?/ 寰呭姞寮?|
| --- | --- | --- | --- |
| 妫€娴嬭繃绋嬬己澶?| 宸插畬鎴愬熀纭€妫€娴?| step_004 `not_observed` | 瀵光€滄憚鍍忓ご鏈鐩栦絾宸插彂鐢熲€濈殑鍒ゆ柇浠嶅急 |
| 鏍规嵁鍓嶅悗鐘舵€佹帹鐞嗙己澶辨楠?| 宸插畬鎴?| 涓棿鏈瀵熸楠ゅ彲鏍囪 inferred_missing | demo step_004 浣嶄簬灏鹃儴锛屽洜姝や笉鑷姩琛ュ叏 |
| 鏍规嵁 SOP 琛ュ叏鍙兘姝ラ | 宸插畬鎴愬熀纭€ SOP 椤哄簭琛ュ叏 | 鍙緭鍑?inferred/skipped/not_observed | 澶嶆潅 SOP 鏉′欢寰幆琛ュ叏闇€瑙勫垯 DSL |
| 鏍规嵁鍘嗗彶瀹為獙琛ュ叏杩囩▼ | 宸叉湁 history model | 鍙粺璁″姩浣滈娆°€佽浆绉绘鐜囥€佹帹鑽?SOP | 鍘嗗彶妯″瀷灏氭湭娣卞害骞跺叆 process 鎺ㄧ悊 |
| 鏍囪琛ュ叏/鐩存帴瑙傚療姝ラ | 宸插畬鎴?| observed/inferred/completed/skipped 瀛楁 | 闇€瑕?UI 鏄庣‘灞曠ず璇佹嵁绛夌骇 |
| 杈撳嚭琛ュ叏鐞嗙敱鍜岀疆淇″害 | 宸插畬鎴?| confidence_reasons + confidence | 浣庣疆淇¤ˉ鍏ㄧ悊鐢卞彲鏇寸粏 |
| 浣庣疆淇¤Е鍙戜汉宸ョ‘璁?| 宸插畬鎴?| step_004 杩涘叆纭闃熷垪 | 鍓嶇瀹℃壒闂幆鏈畬鎴?|

## 鍏€佽瘉鎹摼涓庡璁?

| 鑳藉姏 | 褰撳墠瀹屾垚鎯呭喌 | 瀹屾垚鏁堟灉 | 鏈畬鎴?/ 寰呭姞寮?|
| --- | --- | --- | --- |
| 姝ラ鍏宠仈瑙嗛鐗囨 | 宸插畬鎴?| evidence_refs 鍚?video_event/asset | 闀胯棰戣法娈靛叧鑱旇繕闇€浼樺寲 |
| 姝ラ鍏宠仈鍏抽敭甯?| 宸插畬鎴?| asset refs 鍚?contact/peak/release | 鍏抽敭甯т唬琛ㄦ€ч渶瑙嗚璐ㄩ噺璇勫垎澧炲己 |
| 姝ラ鍏宠仈瀵硅瘽鏂囨湰 | 宸插畬鎴?| state/timeline/context 鍙洖婧?transcript | 鍙ｈ鎸囦唬娑堣В闇€澧炲己 |
| 姝ラ鍏宠仈鏁版嵁搴撹褰?| 缁撴瀯宸叉敮鎸?| database_paths 鍙緭鍏?| demo 娌℃湁鏁版嵁搴撹緭鍏?|
| 姝ラ鍏宠仈鐘舵€佸彉鍖?| 宸插畬鎴?| evidence_refs 鍚?state_change | 鐘舵€佸垎绫昏繕闇€瑙嗚妯″瀷澧炲己 |
| 寤虹珛璇佹嵁閾剧粨鏋?| 宸插畬鎴?| experiment_process.evidence_index | 鍙户缁姞鍥炬暟鎹簱/鍙嶅悜绱㈠紩鎺ュ彛 |
| 鎸夋楠ゅ洖婧瘉鎹?| 宸插畬鎴?| steps[].evidence_refs | 鍓嶇灞曠ず鏈畬鎴?|
| 鎸夎瘉鎹煡鎵炬楠?| 宸插畬鎴?| evidence_index | API 鍖栫▼搴﹂渶澧炲己 |
| 瀹¤鍜屽鐩?| 鍚庣鏁版嵁宸叉敮鎸?| JSONL/JSON 鍙璁?| 鎶ュ憡妯℃澘鍜?UI 澶嶇洏浠嶅彲鍔犲己 |

## 涓冦€丣SON銆佸鍑轰笌妫€绱?

| 鑳藉姏 | 褰撳墠瀹屾垚鎯呭喌 | 瀹屾垚鏁堟灉 | 鏈畬鎴?/ 寰呭姞寮?|
| --- | --- | --- | --- |
| JSON 杈撳嚭鏍煎紡 | 宸插畬鎴?| 澶氫釜 JSON/JSONL artifact | 闇€鍙戝竷 schema 鏂囨。鍜岀増鏈吋瀹圭瓥鐣?|
| 姝ラ绾ц褰曠粨鏋?| 宸插畬鎴?| steps 鍖呭惈鐘舵€併€佺疆淇″害銆佽瘉鎹€佺‘璁ゅ瓧娈?| 鍙姞鍙傛暟鏍￠獙瀛楁 |
| 瀹為獙杩囩▼鏃堕棿绾?| 宸插畬鎴?| `experiment_process_timeline.jsonl` | 鍙笌缁熶竴鏃堕棿绾胯繘涓€姝ュ悎骞跺睍绀?|
| 姝ラ鎽樿 | 宸插畬鎴?| step.name/summary | VLM 鎽樿鍙姞寮?|
| 鍙傛暟淇℃伅 | 宸叉娊鍙?| context parameters=62 | 鍗曚綅褰掍竴鍖栥€佽寖鍥存牎楠屽緟鍔犲己 |
| 缃俊搴?| 宸插畬鎴?| event/step confidence | 寰呭仛璇勬祴闆嗘牎鍑?|
| 璇佹嵁寮曠敤 | 宸插畬鎴?| evidence_refs/asset_refs | 鍙姞 content hash 鍜岄暱鏈熷瓨鍌?URI |
| 琛ュ叏鏍囪 | 宸插畬鎴?| inferred/skipped/not_observed | 鍘嗗彶琛ュ叏鏍囪闇€澧炲己 |
| 寮傚父鏍囪 | 宸插畬鎴?| conflict_flags/anomaly_flags | 寮傚父绫诲瀷搴撻渶鎵╁ぇ |
| 瀵煎嚭鍒版暟鎹簱/鎺ュ彛/鎶ュ憡 | 鏂囦欢鍜屾姤鍛婂凡鏀寔 | JSON/JSONL/Markdown | SQL/API 鍐欏叆鏈畬鎴?|

## 鍏€佽川閲忔鏌ョ粨璁?

| 妫€鏌ラ」 | 褰撳墠缁撹 | 渚濇嵁 |
| --- | --- | --- |
| 鏃堕棿鎴冲榻愭槸鍚﹀噯纭?| 鍩虹鍑嗙‘锛涙敮鎸佹紓绉?寤惰繜鏍″噯缁撴瀯 | demo 娌℃湁澶栭儴 anchor锛寀ser/AI/upload method=none |
| 鍏抽敭鐗囨鎻愬彇鏄惁瀹屾暣 | 鍩虹瀹屾暣 | demo 3 涓?segment銆? 涓?micro |
| 鍏抽敭甯ф槸鍚︿唬琛ㄦ€?| 鍩虹浠ｈ〃鎬?| contact/peak/release anchor 宸茬敓鎴?|
| 鍔ㄤ綔璇嗗埆鏄惁鍑嗙‘ | demo 鍙敤锛岄渶鏇村 GT 璇勬祴 | 3 鏉?action classification |
| 鐘舵€佸彉鍖栨娴嬫槸鍚﹀彲闈?| 鍩虹鍙潬锛岀粏鐘舵€佸緟鍔犲己 | object_state_change=3锛屽鍣?璁惧澶氫负 candidate |
| 姝ラ璇嗗埆鏄惁姝ｇ‘ | demo 鍙敤 | 4 姝ヤ腑 3 瀹屾垚銆? 鏈瀵?|
| 涓嬩竴姝ユ帹鐞嗘槸鍚﹀悎鐞?| 鍩虹鍚堢悊 | next=step_004 |
| 杩囩▼琛ュ叏鏄惁鍙俊 | 涓棿姝ラ鍙ˉ锛屽熬閮ㄦ湭瑙傚療涓嶅己琛?| step_004 not_observed 杩涘叆纭 |
| 璇佹嵁閾炬槸鍚﹀畬鏁?| 鍚庣瀹屾暣 | evidence_refs + evidence_index |
| JSON 鏄惁瑙勮寖 | 宸茶鑼冭緭鍑?| tests 鍜?compileall 閫氳繃 |
| 鏁版嵁鏄惁鍙绱?| 宸叉敮鎸?| vector index + material search |
| 鍘嗗彶杩囩▼鏄惁鍙户鎵垮鐢?| 鍩虹鏀寔 | history model 鏈湴 JSON/JSONL锛涘閮?DB 寰呮帴 |

## 鏈畬鎴?/ 寰呭姞寮轰换鍔′功

### T-01 娑蹭綋娴佸姩涓庢恫浣嶅己纭

- 鐩爣锛氫粠 `liquid_flow_candidate_visual` 鍗囩骇涓哄彲纭鐨勬恫浣撴祦鍔ㄣ€佹恫浣嶅彉鍖栥€佹恫浣撹浆绉昏瘉鎹€?
- 杈撳叆锛氭恫浣?娑查潰/娴佹潫鍒嗗壊妯″瀷鎴栨爣娉ㄦ暟鎹€佸鍣?ROI銆佸叧閿抚搴忓垪銆?
- 杈撳嚭锛歚liquid_flow_confirmed`銆乣liquid_level_change_confirmed`銆佹恫浣嶅彉鍖栭噺銆佽瘉鎹抚銆?
- 楠屾敹锛氬湪鏍囨敞闆嗕笂杈撳嚭 precision/recall/F1锛涗綆缃俊鏍锋湰杩涘叆浜哄伐纭闃熷垪銆?
- Worker A 2026-05-03 update: `advanced_vision_evidence` now gates liquid evidence with `model_inventory.capabilities.liquid_stream_segmentation`; visual levels distinguish `liquid_flow_confirmed` / `liquid_flow_candidate` and `liquid_level_change_confirmed` / `liquid_level_change_candidate`.
- Worker A 2026-05-03 update: keyframe metrics now include explicit `color_profile`, `liquid_level_indicator`, `frame_quality`, `container_state_indicators`, and propagated `quality_limitations`.
- Remaining: no segmentation mask artifact or calibrated container ROI volume estimate is emitted yet; precision/recall/F1 still needs a labeled liquid benchmark.

### T-02 璁惧闈㈡澘 OCR 涓庢寜閽?鏃嬮挳鐘舵€佹ā鍨?

- 鐩爣锛氫粠 `equipment_panel_operation_candidate` 鍗囩骇涓洪潰鏉胯鏁般€佹寜閽€佹棆閽姸鎬佺‘璁ゃ€?
- 杈撳叆锛歜alance/display/button/knob/panel crop 鏍囨敞銆丱CR 寮曟搸鎴栨娴嬫ā鍨嬨€?
- 杈撳嚭锛氳鏁版枃鏈€佹寜閽姸鎬併€佹棆閽搴︺€佹搷浣滃墠鍚庡彉鍖栥€?
- 楠屾敹锛氳鏁?OCR 鍑嗙‘鐜囥€佹寜閽?鏃嬮挳鐘舵€佸垎绫诲噯纭巼銆佸叧鑱旀楠ゆ纭巼銆?

### T-03 瀹瑰櫒寮€鐩?鍏崇洊鐘舵€佸垎绫?

- 鐩爣锛氬埄鐢ㄥ凡鏈?`tube_cap` 绫诲拰鏂板鐘舵€佽鍒欙紝纭 open/closed銆?
- 杈撳叆锛歵ube_cap/bottle_cap 妫€娴嬨€佸鍣?bbox銆佸墠鍚庡抚杞ㄨ抗銆?
- 杈撳嚭锛歚container_opened`銆乣container_closed`銆佺姸鎬佸彉鍖栨椂闂寸偣銆?
- 楠屾敹锛氬紑鐩?鍏崇洊浜嬩欢鍙洖鐜囧拰璇姤鐜囨弧瓒宠瘎娴嬮槇鍊笺€?
- Worker A 2026-05-03 update: `container_open_close` now emits `container_open_close_confirmed` only when cap/lid tokens are present and `model_inventory.capabilities.cap_lid_detection.available` is true; otherwise it remains `container_open_close_candidate`.
- Worker A 2026-05-03 update: metrics include `cap_lid_tokens`, cap/lid model classes, and keyframe state support indicators.
- Remaining: open-vs-close direction is still not classified from explicit before/after cap position tracks.

### T-04 澶氱洰鏍囪韩浠界骇杞ㄨ抗

- 鐩爣锛氶伩鍏嶅悓绫诲鐗╀綋杞ㄨ抗娣锋穯銆?
- 杈撳叆锛歒OLO frame rows銆乼rack_id銆丅yteTrack/DeepSORT/ReID 閫傞厤銆?
- 杈撳嚭锛氱ǔ瀹?object_track_id銆佺Щ鍔ㄨ矾寰勩€佸仠鐣?鎷胯捣/鏀句笅浜嬩欢銆?
- 楠屾敹锛氬悓绫诲鐩爣瑙嗛涓建杩?ID switch 闄嶄綆鍒板彲鎺ュ彈鑼冨洿銆?
- Worker A 2026-05-03 update: trajectory evidence metrics now include `track_id`, `track_source`, `identity_source`, `identity_confidence`, `same_class_track_count`, `max_same_class_per_frame`, explicit/inferred track counts, and `identity_risk_reasons`.
- Worker A 2026-05-03 update: same-class multi-target inferred tracks are downgraded to `trajectory_candidate_identity_risk` and require human confirmation.
- Remaining: no ByteTrack/DeepSORT/ReID adapter is implemented in this task; the current fallback remains bbox temporal association.

### T-05 鍔ㄤ綔鍒嗙被妯″瀷鎴?VLM 浜屾鍒ゅ埆

- 鐩爣锛氭彁鍗?weighing/pipetting/sample_handling/recording 绛夊鏉傚姩浣滆瘑鍒噯纭害銆?
- 杈撳叆锛歮icro clips銆佸叧閿抚銆佸璞¤建杩广€佽闊?鏂囨湰涓婁笅鏂囥€?
- 杈撳嚭锛氬姩浣滅被鍒€佸姩浣滈樁娈点€佺疆淇″害鍜岃В閲娿€?
- 楠屾敹锛氬姩浣滃垎绫诲噯纭巼銆佹贩娣嗙煩闃点€佷綆缃俊鑷姩杩涘叆纭銆?

### T-06 SOP 瑙勫垯 DSL

- 鐩爣锛氭敮鎸佸鏉傚垎鏀€佹潯浠跺惊鐜€佷簰鏂ャ€佽秴鏃躲€佸弬鏁扮害鏉熷拰鍐茬獊娑堣В銆?
- 杈撳叆锛歋OP DSL/JSON銆佸弬鏁版潯浠躲€佸巻鍙茬粺璁°€?
- 杈撳嚭锛氱姸鎬佹満鎵ц鍥俱€佸綋鍓嶆楠ゃ€佷笅涓€姝ャ€佸啿绐佺悊鐢便€?
- 楠屾敹锛氳鐩栬烦姝ャ€侀噸澶嶃€佸洖閫€銆佸苟琛屽垎鏀€佸鏉′欢寰幆娴嬭瘯闆嗐€?

### T-07 鍘嗗彶瀹為獙妯″瀷骞跺叆姝ラ鎺ㄧ悊

- 鐩爣锛氳鍘嗗彶缁熻涓嶄粎鐢熸垚鎶ュ憡锛岃繕鐩存帴褰卞搷缂哄け琛ュ叏鍜屼笅涓€姝ユ帹鐞嗐€?
- 杈撳叆锛歨istory model銆佸綋鍓?process銆丼OP銆?
- 杈撳嚭锛氬巻鍙插厛楠岀疆淇″害銆佸父瑙佹楠ゅ簭鍒椼€佸紓甯稿亸绂荤▼搴︺€?
- 楠屾敹锛氳ˉ鍏ㄦ楠ゅ繀椤绘樉绀哄巻鍙蹭緷鎹拰璐＄尞鍒嗘暟銆?

### T-08 澶栭儴鏁版嵁搴?/ LIMS / Vector DB 鎺ュ叆

- 鐩爣锛氳 context_fusion 妫€绱㈢湡瀹炴暟鎹簱锛岃€屼笉鏄彧璇绘湰鍦?JSON/JSONL銆?
- 杈撳叆锛歋QL銆丩IMS銆乿ector DB銆佸疄楠岃褰?API銆?
- 杈撳嚭锛歞atabase_refs銆佹潗鏂欐壒娆°€佸巻鍙插弬鏁般€丼OP 鐗堟湰銆?
- 楠屾敹锛氭楠よ瘉鎹摼鍙拷婧埌鏁版嵁搴撹褰?ID銆?

### T-09 浜哄伐纭鍓嶇瀹℃壒娴?

- 鐩爣锛氭妸 `human_confirmation_queue.jsonl` 鎺ュ埌 UI銆?
- 杈撳叆锛氱‘璁ら槦鍒椼€佽瘉鎹?refs銆佸叧閿抚鍜岀墖娈点€?
- 杈撳嚭锛歛pproved/rejected/needs_review 鍐崇瓥鍐欏洖銆?
- 楠屾敹锛氬墠绔彲鎸夋楠ゅ鎵癸紝鍚庣 `experiment_process.json` 鐘舵€佸悓姝ユ洿鏂般€?

### T-10 鏃堕棿鏍″噯鐪熷疄 anchor 璇勬祴

- 鐩爣锛氳瘎浼拌棰戙€佽闊炽€佺敤鎴锋枃鏈€丄I 鍥炲銆佷笂浼犲浘鏂囦箣闂寸殑鐪熷疄婕傜Щ鍜屽欢杩熴€?
- 杈撳叆锛氱湡瀹?user_events銆乤i_events銆乽ploads銆乧alibration anchors銆?
- 杈撳嚭锛氭瘡婧?slope/intercept/latency/residual銆?
- 楠屾敹锛氭畫宸粺璁″彲鎶ュ憡锛岃秴闃堝€兼簮鑷姩鏍囪寮傚父銆?

### T-11 鍏抽敭鐗囨涓庡叧閿抚璐ㄩ噺璇勪及

- 鐩爣锛氬垽鏂叧閿墖娈垫槸鍚﹀畬鏁淬€佸叧閿抚鏄惁浠ｈ〃鎺ヨЕ/宄板€?閲婃斁銆?
- 杈撳叆锛歮icro clips銆乲eyframes銆乊OLO rows銆佷汉宸?GT銆?
- 杈撳嚭锛歝overage銆乺epresentativeness銆乥lur/occlusion/visibility score銆?
- 楠屾敹锛氫綆璐ㄩ噺鍏抽敭甯ц嚜鍔ㄨˉ鎶芥垨杩涘叆澶嶆牳銆?

### T-12 JSON Schema 涓庢帴鍙ｅ绾?

- 鐩爣锛氬皢褰撳墠 JSON/JSONL 浜х墿鍥哄寲涓虹増鏈寲 schema銆?
- 杈撳叆锛歷ideo_understanding銆乪xperiment_context銆乪xperiment_process銆乤sset catalog銆?
- 杈撳嚭锛歴chema 鏂囦欢銆佸瓧娈佃鏄庛€佸吋瀹规祴璇曘€?
- 楠屾敹锛欳I 涓牎楠屾墍鏈?artifact schema銆?

### T-13 杩囩▼瀹¤鎶ュ憡涓庡鐩?UI

- 鐩爣锛氭妸璇佹嵁閾俱€佸紓甯搞€佽ˉ鍏ㄣ€佷汉宸ョ‘璁よ浆鎴愬璁?澶嶇洏椤甸潰鎴栨姤鍛娿€?
- 杈撳叆锛歱rocess銆乼imeline銆乤ssets銆乧onfirmation decisions銆?
- 杈撳嚭锛氭楠ょ骇澶嶇洏鎶ュ憡銆佽瘉鎹储寮曘€佸紓甯歌鏄庛€?
- 楠屾敹锛氬璁′汉鍛樺彲鎸夋楠ゆ垨璇佹嵁鍙屽悜杩芥函銆?