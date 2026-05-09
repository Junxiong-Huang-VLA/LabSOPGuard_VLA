import { Navigate, useParams } from 'react-router-dom'

export default function VideoAnalysis() {
  const { id } = useParams()
  return <Navigate to={id ? `/experiments/${id}/workspace` : '/experiments'} replace />
}
