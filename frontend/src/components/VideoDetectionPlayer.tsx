import { useRef, useState } from 'react';

interface Props {
  experimentId: string;
  streamType?: 'first_person' | 'third_person';
}

export default function VideoDetectionPlayer({ experimentId, streamType = 'first_person' }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [activeStream, setActiveStream] = useState(streamType);

  const videoSrc = `/api/v1/experiments/${experimentId}/files/key_action_index/clips/experiment_focus/${activeStream}_yolo_annotated.mp4`;

  return (
    <div className="bg-white rounded-lg border p-4 mb-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">YOLO Detection Replay</h3>
      <div className="relative aspect-video bg-black rounded overflow-hidden">
        <video
          ref={videoRef}
          src={videoSrc}
          className="w-full h-full object-contain"
          onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime || 0)}
          onLoadedMetadata={() => setDuration(videoRef.current?.duration || 0)}
          controls
        />
      </div>
      <div className="mt-2 flex items-center gap-2 text-xs text-gray-500">
        <span>{formatTime(currentTime)}</span>
        <input
          type="range"
          min={0}
          max={duration}
          step={0.1}
          value={currentTime}
          onChange={(e) => {
            const t = parseFloat(e.target.value);
            if (videoRef.current) videoRef.current.currentTime = t;
            setCurrentTime(t);
          }}
          className="flex-1 h-1"
        />
        <span>{formatTime(duration)}</span>
        <select
          className="ml-2 border rounded px-1 py-0.5 text-xs"
          value={activeStream}
          onChange={(e) => setActiveStream(e.target.value as any)}
        >
          <option value="first_person">First Person</option>
          <option value="third_person">Third Person</option>
        </select>
      </div>
    </div>
  );
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
