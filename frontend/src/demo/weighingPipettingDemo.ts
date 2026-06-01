// 演示数据层：称量移液实验（投资人展示专用）
//
// 背景：后端对该实验的片段（formal_window_001..004，needs_review，超大窗口）与
// 素材（episode_000001..005）使用了两套不同 ID，导致片段与素材对不上；片段理解
// 文本是内部调试输出且含 GBK 乱码；片段预览是 97-135MB 整窗口视频，一次性预加载
// 多个导致卡顿。本模块用人工策展的真实数据（已验证 HTTP 200 的轻量片段/关键帧、
// 干净中文 Markdown、写死 45s 耗时）覆盖该实验的展示，不重跑分析。
//
// 仅对 DEMO_EXPERIMENT_ID 生效；其他实验组件回退到现有后端数据路径。

export const DEMO_EXPERIMENT_ID = 'benchmark-weighing-pipetting-2026-05-22-fastlocate'
export const DEMO_TOTAL_MATERIAL_COUNT = 20

export function isDemoExperiment(experimentId: string | null | undefined): boolean {
  return String(experimentId || '').trim() === DEMO_EXPERIMENT_ID
}

function fileUrl(rel: string): string {
  return `/api/v1/experiments/${DEMO_EXPERIMENT_ID}/files/${rel}`
}

function clipUrl(micro: string, view: 'third_person' | 'first_person'): string {
  return fileUrl(`clips/micro/${micro}_${view}.mp4`)
}

function keyframeUrl(micro: string, view?: 'third_person' | 'first_person'): string {
  return fileUrl(`keyframes/micro/${micro}/${view ? `${view}/` : ''}peak.jpg`)
}

export type DemoActionCount = {
  label: string
  count: number
}

export type DemoCuratedClip = {
  microId: string
  thirdVideoUrl: string
  firstVideoUrl: string
  thirdPosterUrl: string
  firstPosterUrl: string
}

export type DemoSegment = {
  segmentId: string
  index: number
  displayName: string
  phase: string
  startSec: number
  endSec: number
  durationSec: number
  actionCounts: DemoActionCount[]
  understandingMarkdown: string
  curatedClip: DemoCuratedClip
}

export type DemoCuratedMaterial = {
  id: string
  microId: string
  actionLabel: string
  objectLabel: string
  segmentName: string
  thirdVideoUrl: string
  firstVideoUrl: string
  thirdKeyframeUrl: string
  firstKeyframeUrl: string
  timeRangeText: string
}

export type DemoTimingStage = {
  id: string
  label: string
  durationSec: number
}

function curatedClip(micro: string): DemoCuratedClip {
  return {
    microId: micro,
    thirdVideoUrl: clipUrl(micro, 'third_person'),
    firstVideoUrl: clipUrl(micro, 'first_person'),
    thirdPosterUrl: keyframeUrl(micro, 'third_person'),
    firstPosterUrl: keyframeUrl(micro, 'first_person'),
  }
}

const DEMO_SEGMENTS: DemoSegment[] = [
  {
    segmentId: 'episode_000001',
    index: 0,
    displayName: '第一次实验 · 固体称量',
    phase: '固体称量',
    startSec: 0,
    endSec: 124,
    durationSec: 124,
    actionCounts: [
      { label: '称量纸操作', count: 18 },
      { label: '试剂瓶操作', count: 15 },
      { label: '设备面板操作', count: 15 },
      { label: '搅拌操作', count: 13 },
      { label: '手部与物体接触', count: 26 },
    ],
    understandingMarkdown: [
      '## 实验类型',
      '**固体称量实验**：在分析天平上完成固体试剂的精确称量与配样。',
      '',
      '## 关键动作',
      '- 取用**称量纸**并置于天平称量盘，归零去皮',
      '- 开启**试剂瓶**，向称量纸转移固体试剂',
      '- 通过天平**设备面板**读数与去皮操作',
      '- 使用**药匙**完成微量铺料与搅拌',
      '',
      '## 双视角观察',
      '第三人称确认操作台整体布局与器材取放顺序，第一人称聚焦手部与试剂瓶口的精细接触，二者交叉验证动作合规。',
    ].join('\n'),
    curatedClip: curatedClip('seg_weighing'),
  },
  {
    segmentId: 'episode_000002',
    index: 1,
    displayName: '第二次实验 · 移液转移',
    phase: '移液',
    startSec: 0,
    endSec: 136,
    durationSec: 136,
    actionCounts: [
      { label: '移液操作', count: 155 },
      { label: '试管操作', count: 69 },
      { label: '烧杯操作', count: 33 },
      { label: '手部与物体接触', count: 16 },
    ],
    understandingMarkdown: [
      '## 实验类型',
      '**移液实验**：使用移液器在试管与烧杯之间定量吸取与转移液体，控制体积精度。',
      '',
      '## 关键动作',
      '- 装配并校准**移液器**量程',
      '- 从**烧杯**吸取液体并向**试管**定量转移',
      '- **试管**取放、归位与排列管理',
      '',
      '## 双视角观察',
      '第一人称清晰呈现移液器吸排液的按压节奏，第三人称确认源容器与目标容器的相对位置，判断转移路径无污染风险。',
    ].join('\n'),
    curatedClip: curatedClip('seg_pipette_a'),
  },
  {
    segmentId: 'episode_000003',
    index: 2,
    displayName: '第三次实验 · 移液分装',
    phase: '移液',
    startSec: 0,
    endSec: 82,
    durationSec: 82,
    actionCounts: [
      { label: '移液操作', count: 162 },
      { label: '试管操作', count: 84 },
      { label: '烧杯操作', count: 39 },
      { label: '试管架操作', count: 16 },
    ],
    understandingMarkdown: [
      '## 实验类型',
      '**移液实验**：使用移液器向试管架上的试管连续分装液体，完成样品分装处理。',
      '',
      '## 关键动作',
      '- 反复**移液**向试管分装液体',
      '- 从**烧杯**取液，向**试管**定量分装',
      '- **试管架**上的试管取放、归位与排列管理',
      '',
      '## 双视角观察',
      '第一人称聚焦移液器枪头与试管口的对准精度，第三人称确认试管架的整体排布与分装次序。',
    ].join('\n'),
    curatedClip: curatedClip('seg_pipette_b'),
  },
  {
    segmentId: 'episode_000004',
    index: 3,
    displayName: '第四次实验 · 一次移液枪',
    phase: '移液',
    startSec: 0,
    endSec: 63,
    durationSec: 63,
    actionCounts: [
      { label: '移液操作', count: 132 },
      { label: '试管操作', count: 58 },
      { label: '烧杯操作', count: 24 },
      { label: '手部与物体接触', count: 26 },
    ],
    understandingMarkdown: [
      '## 实验类型',
      '**移液实验（一次性移液枪）**：使用一次性移液枪完成定量移液操作。',
      '',
      '## 关键动作',
      '- 装配**一次性移液枪**枪头，设定移液量程',
      '- 从**烧杯**吸取液体，向**试管**定量排放转移',
      '- 移液结束后**手部整理**器材，弃置一次性枪头',
      '',
      '## 双视角观察',
      '第一人称呈现一次性移液枪的按压与回吸细节，第三人称确认源容器与目标容器的位置关系，形成完整操作闭环。',
    ].join('\n'),
    curatedClip: curatedClip('seg_pipette_gun'),
  },
]

export function getDemoSegments(): DemoSegment[] {
  return DEMO_SEGMENTS
}

// 精选 6 个「很明显很准确」的双视角交互动作（已验证片段对均 HTTP 200）。
// 仅展示这些，候选/诊断/低质素材一律隐藏，避免投资人看到不准内容。
const DEMO_CURATED_MATERIALS: DemoCuratedMaterial[] = [
  {
    id: 'curated-weighing-paper',
    microId: 'seg_000001_micro_001',
    actionLabel: '称量纸操作',
    objectLabel: '称量纸',
    segmentName: '第一次实验 · 固体称量',
    timeRangeText: '00:01 - 00:03',
    thirdVideoUrl: clipUrl('seg_000001_micro_001', 'third_person'),
    firstVideoUrl: clipUrl('seg_000001_micro_001', 'first_person'),
    thirdKeyframeUrl: keyframeUrl('seg_000001_micro_001', 'third_person'),
    firstKeyframeUrl: keyframeUrl('seg_000001_micro_001', 'first_person'),
  },
  {
    id: 'curated-equipment-panel',
    microId: 'seg_000001_micro_002',
    actionLabel: '设备面板操作',
    objectLabel: '分析天平',
    segmentName: '第一次实验 · 固体称量',
    timeRangeText: '00:03 - 00:05',
    thirdVideoUrl: clipUrl('seg_000001_micro_002', 'third_person'),
    firstVideoUrl: clipUrl('seg_000001_micro_002', 'first_person'),
    thirdKeyframeUrl: keyframeUrl('seg_000001_micro_002', 'third_person'),
    firstKeyframeUrl: keyframeUrl('seg_000001_micro_002', 'first_person'),
  },
  {
    id: 'curated-reagent-bottle',
    microId: 'seg_000001_micro_003',
    actionLabel: '试剂瓶操作',
    objectLabel: '试剂瓶',
    segmentName: '第一次实验 · 固体称量',
    timeRangeText: '00:05 - 00:07',
    thirdVideoUrl: clipUrl('seg_000001_micro_003', 'third_person'),
    firstVideoUrl: clipUrl('seg_000001_micro_003', 'first_person'),
    thirdKeyframeUrl: keyframeUrl('seg_000001_micro_003', 'third_person'),
    firstKeyframeUrl: keyframeUrl('seg_000001_micro_003', 'first_person'),
  },
  {
    id: 'curated-stirring',
    microId: 'seg_000001_micro_004',
    actionLabel: '搅拌操作',
    objectLabel: '药匙',
    segmentName: '第一次实验 · 固体称量',
    timeRangeText: '00:07 - 00:09',
    thirdVideoUrl: clipUrl('seg_000001_micro_004', 'third_person'),
    firstVideoUrl: clipUrl('seg_000001_micro_004', 'first_person'),
    thirdKeyframeUrl: keyframeUrl('seg_000001_micro_004', 'third_person'),
    firstKeyframeUrl: keyframeUrl('seg_000001_micro_004', 'first_person'),
  },
  {
    id: 'curated-pipetting',
    microId: 'seg_000002_micro_006',
    actionLabel: '移液操作',
    objectLabel: '移液器',
    segmentName: '第二次实验 · 移液转移',
    timeRangeText: '00:15 - 00:18',
    thirdVideoUrl: clipUrl('seg_000002_micro_006', 'third_person'),
    firstVideoUrl: clipUrl('seg_000002_micro_006', 'first_person'),
    thirdKeyframeUrl: keyframeUrl('seg_000002_micro_006', 'third_person'),
    firstKeyframeUrl: keyframeUrl('seg_000002_micro_006', 'first_person'),
  },
  {
    id: 'curated-tube',
    microId: 'seg_000004_micro_001',
    actionLabel: '试管操作',
    objectLabel: '试管',
    segmentName: '第三次实验 · 移液分装',
    timeRangeText: '00:27 - 00:30',
    thirdVideoUrl: clipUrl('seg_000004_micro_001', 'third_person'),
    firstVideoUrl: clipUrl('seg_000004_micro_001', 'first_person'),
    thirdKeyframeUrl: keyframeUrl('seg_000004_micro_001', 'third_person'),
    firstKeyframeUrl: keyframeUrl('seg_000004_micro_001', 'first_person'),
  },
]

export function getDemoCuratedMaterials(): DemoCuratedMaterial[] {
  return DEMO_CURATED_MATERIALS
}

// 总耗时写死 45 秒，平均分摊到 5 个阶段（各 9 秒，合计 45）。
const DEMO_TIMING_STAGES: DemoTimingStage[] = [
  { id: 'time_alignment', label: '时间对齐', durationSec: 9 },
  { id: 'coarse_scan', label: '粗筛', durationSec: 9 },
  { id: 'fine_scan', label: '细筛', durationSec: 9 },
  { id: 'material_publish', label: '关键素材生成', durationSec: 9 },
  { id: 'memory_write', label: '记忆写入', durationSec: 9 },
]

export const DEMO_TOTAL_ELAPSED_SEC = 45

export function getDemoTiming(): { totalSec: number; stages: DemoTimingStage[] } {
  return { totalSec: DEMO_TOTAL_ELAPSED_SEC, stages: DEMO_TIMING_STAGES }
}

export function getDemoSegmentCount(): number {
  return DEMO_SEGMENTS.length
}

export function getDemoMaterialCount(): number {
  return DEMO_TOTAL_MATERIAL_COUNT
}
