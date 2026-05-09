import { Navigate, useParams } from 'react-router-dom'

export default function StepReview() {
  const { id, stepId } = useParams()
  return <Navigate to={id && stepId ? `/experiments/${id}/steps/${stepId}` : '/experiments'} replace />
}
