import { useEffect, useState } from 'react';
import axios from 'axios';

interface PendingItem {
  event_id: string;
  event_type: string;
  display_name: string;
  start_time_sec: number;
  end_time_sec: number;
  confidence: number;
  evidence_grade: string;
  has_clip: boolean;
  keyframe_count: number;
}

interface Props {
  experimentId: string;
}

export default function PendingMaterials({ experimentId }: Props) {
  const [pending, setPending] = useState<PendingItem[]>([]);
  const [approvedCount, setApprovedCount] = useState(0);
  const [loading, setLoading] = useState(true);

  const fetchPending = () => {
    axios.get(`/api/v1/experiments/${experimentId}/materials/pending-review`)
      .then(res => {
        setPending(res.data.pending || []);
        setApprovedCount(res.data.approved_count || 0);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchPending(); }, [experimentId]);

  const handleApprove = (eventId: string) => {
    axios.post(`/api/v1/experiments/${experimentId}/materials/approve`, { event_id: eventId })
      .then(() => fetchPending());
  };

  const handleReject = (eventId: string) => {
    axios.post(`/api/v1/experiments/${experimentId}/materials/reject`, { event_id: eventId, reason: 'manual_reject' })
      .then(() => fetchPending());
  };

  const handleApproveAll = () => {
    axios.post(`/api/v1/experiments/${experimentId}/materials/approve-all`)
      .then(() => fetchPending());
  };

  if (loading) return null;
  if (pending.length === 0 && approvedCount === 0) return null;

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-white rounded-lg border p-4 mb-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700">
          Material Review ({pending.length} pending, {approvedCount} approved)
        </h3>
        {pending.length > 0 && (
          <button
            onClick={handleApproveAll}
            className="text-xs bg-green-500 text-white px-3 py-1 rounded hover:bg-green-600"
          >
            Approve All
          </button>
        )}
      </div>

      {pending.length === 0 ? (
        <div className="text-sm text-gray-400">All materials reviewed.</div>
      ) : (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {pending.map(item => (
            <div key={item.event_id} className="flex items-center gap-3 p-2 border rounded hover:bg-gray-50">
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{item.display_name}</div>
                <div className="text-xs text-gray-500">
                  {item.event_type} | {formatTime(item.start_time_sec)}-{formatTime(item.end_time_sec)} | conf={item.confidence.toFixed(2)}
                </div>
                <div className="text-xs text-gray-400">
                  {item.has_clip ? 'clip' : ''} {item.keyframe_count} frames | {item.evidence_grade}
                </div>
              </div>
              <button
                onClick={() => handleApprove(item.event_id)}
                className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded hover:bg-green-200"
              >
                Approve
              </button>
              <button
                onClick={() => handleReject(item.event_id)}
                className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded hover:bg-red-200"
              >
                Reject
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
