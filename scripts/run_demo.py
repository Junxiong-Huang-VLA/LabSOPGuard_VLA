from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from experiment.service import ExperimentService
from labsopguard.workflow import FormalExperimentWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the formal LabSOPGuard experiment pipeline')
    parser.add_argument('--video', required=True)
    parser.add_argument('--title', default='LabSOPGuard Demo')
    parser.add_argument('--context', default='')
    parser.add_argument('--protocol', default='')
    args = parser.parse_args()

    exp_id = f'demo_{Path(args.video).stem}'
    service = ExperimentService(
        vlm_api_key=os.environ.get('DASHSCOPE_API_KEY'),
        vlm_base_url=os.environ.get('DASHSCOPE_BASE_URL'),
        frame_sample_interval=2.0,
        max_frames=24,
    )
    service.set_video(args.video)
    service.set_context(args.context)
    service.set_protocol(args.protocol)
    result = service.process(experiment_id=exp_id, experiment_title=args.title)
    saved = service.save_outputs()
    structured = FormalExperimentWorkflow().build_structured_output(
        {
            'experiment_id': exp_id,
            'title': args.title,
            'video_paths': [args.video],
            'context_inputs': [{'text': args.context}] if args.context else [],
            'protocol_text': args.protocol,
        },
        result,
    )
    structured_path = Path(saved['experiment']).parent / 'structured.json'
    structured_path.write_text(__import__('json').dumps(structured, ensure_ascii=False, indent=2), encoding='utf-8')
    print(Path(saved['experiment']).parent)
    print(structured_path)


if __name__ == '__main__':
    main()
