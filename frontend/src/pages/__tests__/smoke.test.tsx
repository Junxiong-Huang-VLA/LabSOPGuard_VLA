import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { EmptyEvidence } from '../../components/EvidenceUI'

describe('frontend smoke', () => {
  it('renders shared evidence UI', () => {
    render(<EmptyEvidence title="证据工作台" description="smoke" />)
    expect(screen.getByText('证据工作台')).toBeInTheDocument()
    expect(screen.getByText('smoke')).toBeInTheDocument()
  })
})
