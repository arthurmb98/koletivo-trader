import { useEffect, useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { brl, cn, num, pct } from '@/lib/utils'
import type { BankKey, CaseKey, Parecer, ParecerCaseAvg, PeriodRow, RunSide, StudyFile, Winner } from '@/lib/types'

const SIGNAL_CASES: { key: CaseKey; label: string; help: string }[] = [
  {
    key: 'last_candles',
    label: 'Últimos candles',
    help: 'Operação sempre no M5. Os 15 minutos em M1 só montam o tensor da estratégia dos últimos candles de 5 min.',
  },
]
const CASE_HELP = Object.fromEntries(SIGNAL_CASES.map((c) => [c.key, c.help])) as Record<CaseKey, string>
const BANKS: BankKey[] = ['500', '1000', '5000', '10000']

function listWinners(data: StudyFile | null, bank: BankKey): Winner[] {
  const node = data?.winners?.last_candles?.[bank]
  if (!node) return []
  if (Array.isArray(node)) {
    return node.filter((winner) => winner.params.data.timeframe !== 'm1')
  }
  return node.m5 ?? []
}

function setupLabel(winner: Winner) {
  const risk = winner.params.risk
  const exe = winner.params.execution
  const dir = exe.direction === 'fade' ? 'contra a previsão' : 'seguir'
  const trail = risk.trailing_enabled ? ' · trailing' : ''
  return `5 min · últimos candles · ${dir}${trail}`
}

function stopGainOf(winner: Winner) {
  return {
    stop: Number(winner.params.risk.stop_points),
    gain: Number(winner.params.risk.gain_points),
  }
}

function stopGainFromLabel(label: string | null | undefined) {
  const match = label?.match(/(\d+)\s*\/\s*(\d+)/)
  if (!match) return null
  return { stop: Number(match[1]), gain: Number(match[2]) }
}

function StopGainMark({
  stop,
  gain,
  className,
}: {
  stop: number
  gain: number
  className?: string
}) {
  if (!Number.isFinite(stop) || !Number.isFinite(gain)) return null
  return (
    <p className={cn('text-[10px] tabular-nums tracking-wide text-muted-foreground/55', className)}>
      stop {stop} · gain {gain}
    </p>
  )
}

function avgMonth(side: RunSide | Winner | undefined) {
  if (!side?.metrics) return 0
  const months = Math.max(side.by_period?.monthly?.length ?? 0, 1)
  return side.metrics.net_pnl / months
}

function lotSide(winner: Winner | undefined, scaled: boolean) {
  if (!winner) return undefined
  if (scaled) return winner.lot_scaled ?? winner.linear ?? winner
  return winner.lot_fixed ?? winner.one_contract ?? winner
}

function LotCol({ label, side, hint }: { label: string; side: RunSide | Winner | undefined; hint?: string }) {
  if (!side?.metrics) return null
  const monthly = avgMonth(side)
  return (
    <div className="rounded-xl border border-border/70 bg-card/40 p-3">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className={cn('mt-2 font-display text-2xl font-bold', monthly >= 0 ? 'text-gain' : 'text-loss')}>{brl(monthly)}</p>
      <p className="mt-1 text-xs text-muted-foreground">
        /mês · total {brl(side.metrics.net_pnl)} · tombo {pct(side.metrics.max_drawdown_pct)}
        {side.metrics.max_contracts && side.metrics.max_contracts > 1 ? ` · até ${side.metrics.max_contracts} minis` : ''}
      </p>
      {hint ? <p className="mt-2 text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  )
}

function formatDecision(value: unknown) {
  if (value === 'ml_guard') return 'ML + guarda dos últimos candles'
  if (value === 'price_action_ml') return 'padrões definem o lado'
  return 'só o modelo no último candle'
}

function Kpi({
  label,
  value,
  hint,
  positive,
}: {
  label: string
  value: string
  hint?: string
  positive?: boolean | null
}) {
  return (
    <div className="rounded-2xl border border-border bg-elevated/70 p-4">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p
        className={cn(
          'mt-2 font-display text-2xl font-bold',
          positive === true && 'text-gain',
          positive === false && 'text-loss',
        )}
      >
        {value}
      </p>
      {hint ? <p className="mt-1 text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  )
}

function PeriodBars({ title, rows }: { title: string; rows: PeriodRow[] }) {
  const data = rows.length > 120 ? rows.filter((_, i) => i % Math.ceil(rows.length / 120) === 0) : rows
  return (
    <div className="rounded-2xl border border-border bg-elevated/40 p-4">
      <h3 className="font-display font-semibold">{title}</h3>
      <div className="mt-4 h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
            <XAxis dataKey="t" hide />
            <YAxis tick={{ fill: '#a1a1aa', fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
              formatter={(value) => brl(Number(value ?? 0))}
            />
            <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
              {data.map((row) => (
                <Cell key={row.t} fill={row.pnl >= 0 ? '#34d399' : '#fb7185'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function WinnerCard({
  winner,
  title,
  active,
  onClick,
}: {
  winner: Winner
  title: string
  active: boolean
  onClick: () => void
}) {
  const fixed = lotSide(winner, false)
  const scaled = lotSide(winner, true)
  const monthly = avgMonth(fixed)
  const { stop, gain } = stopGainOf(winner)
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'relative rounded-2xl border p-5 text-left transition-colors',
        active ? 'border-primary bg-card' : 'border-border bg-elevated/50 hover:bg-card/80',
      )}
    >
      <StopGainMark stop={stop} gain={gain} className="absolute right-4 top-4" />
      <p className="pr-28 text-sm text-muted-foreground">
        {title}
        {monthly < 0 ? ' · prejuízo no teste' : ''}
      </p>
      <p className="mt-1 font-display text-lg font-semibold">{setupLabel(winner)}</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <LotCol label="1 mini" side={fixed} hint="Ranking" />
        <LotCol label="Lote que sobe" side={scaled} hint="Mesmo setup, mais minis" />
      </div>
    </button>
  )
}

function SetupParams({ winner }: { winner: Winner }) {
  const risk = winner.params.risk
  const filters = winner.params.filters
  const exe = winner.params.execution
  const rows: [string, string][] = [
    ['Tempo gráfico', '5 minutos'],
    ['Como decide', formatDecision(exe.decision)],
    ['Direção', exe.direction === 'fade' ? 'contra a previsão' : 'seguir a previsão'],
    ['Stop', `${risk.stop_points} pts`],
    ['Alvo', `${risk.gain_points} pts`],
    ['Trailing', risk.trailing_enabled ? `sim, após ${risk.trailing_trigger_points} pts` : 'não'],
    ['Stop diário', Number(risk.daily_loss_points) > 0 ? `${risk.daily_loss_points} pts` : 'desligado'],
    ['Máx. ops / dia', String(risk.max_trades_per_day)],
    ['Horário-ouro', filters.gold_hours_only ? 'sim' : 'não'],
    ['Gap máximo', filters.max_gap_points == null ? 'sem limite' : `${filters.max_gap_points} pts`],
  ]
  return (
    <div className="rounded-2xl border border-border bg-elevated/50 p-4">
      <h4 className="font-display text-sm font-semibold text-primary">O que muda neste setup</h4>
      <dl className="mt-3 grid gap-2 sm:grid-cols-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-start justify-between gap-4 text-sm">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="text-right font-medium">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-xs text-muted-foreground">
        Igual em todos: sessão 9h15–17h, sem almoço, fill na abertura seguinte, custo R$ 1 por operação.
      </p>
    </div>
  )
}

function CaseCompare({
  parecer,
  lookback,
  onPick,
}: {
  parecer: Parecer
  lookback: { m1: number; m5: number }
  onPick: (bank?: BankKey) => void
}) {
  const rows = parecer.by_case?.length
    ? parecer.by_case
    : parecer.monthly.reduce<ParecerCaseAvg[]>((acc, row) => {
        if (acc.some((item) => item.bank === row.bank)) return acc
        acc.push(row)
        return acc
      }, [])
  const hit = parecer.ml_hit.daytrade ?? parecer.ml_hit.m1 ?? 0

  return (
    <section id="parecer" className="relative mx-auto max-w-6xl px-5 py-10 sm:px-8">
      <p className="text-sm uppercase tracking-[0.2em] text-primary">Comparar</p>
      <h2 className="mt-2 font-display text-3xl font-bold">{parecer.headline}</h2>
      <p className="mt-3 max-w-3xl text-muted-foreground">
        Número grande = P&amp;L no teste da banca.{' '}
        {parecer.n_months_note} Acerto do daytrade: {pct(hit)}. Os {lookback.m1} candles de 1 min entram só como
        contexto dos últimos {lookback.m5} de 5 min. {parecer.dd_floor}
      </p>
      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {BANKS.map((bank) => {
          const row = rows.find((item) => String(item.bank) === bank)
          const sg = stopGainFromLabel(row?.label)
          return (
            <button
              key={bank}
              type="button"
              onClick={() => onPick(bank)}
              className="relative rounded-2xl border border-border bg-elevated/50 p-5 text-left transition-colors hover:border-primary"
            >
              {sg ? <StopGainMark stop={sg.stop} gain={sg.gain} className="absolute right-4 top-4" /> : null}
              <p className="text-xs uppercase tracking-wide text-muted-foreground">{brl(Number(bank))}</p>
              {row?.avg_fixed == null ? (
                <p className="mt-3 text-sm text-muted-foreground">Sem resultado nesta banca.</p>
              ) : (
                <>
                  <p className={cn('mt-3 font-display text-2xl font-bold', row.avg_fixed >= 0 ? 'text-gain' : 'text-loss')}>
                    {brl(row.avg_fixed)}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">M5 · últimos candles</p>
                </>
              )}
            </button>
          )
        })}
      </div>
      <ul className="mt-8 space-y-2 text-sm text-foreground/90">
        {parecer.strategy.map((item) => (
          <li key={item}>· {item}</li>
        ))}
      </ul>
    </section>
  )
}

export function StudyPage() {
  const [data, setData] = useState<StudyFile | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [bank, setBank] = useState<BankKey>('1000')
  const [pick, setPick] = useState(0)
  const [chartScaled, setChartScaled] = useState(false)
  const [showIntra, setShowIntra] = useState(false)

  useEffect(() => {
    const load = async () => {
      const bust = `v=${Date.now()}`
      const urls = [`/studies.json?${bust}`, `/api/studies?${bust}`]
      for (const url of urls) {
        try {
          const res = await fetch(url, { cache: 'no-store' })
          if (!res.ok) continue
          const json = (await res.json()) as StudyFile
          const winnerKeys = Object.keys(json.winners ?? {})
          if (winnerKeys.length && !winnerKeys.includes('last_candles')) {
            continue
          }
          setData(json)
          const firstBank = String(json.banks?.[1] ?? json.banks?.[0] ?? 1000) as BankKey
          setBank(BANKS.includes(firstBank) ? firstBank : '1000')
          setChartScaled(false)
          return
        } catch {
          /* try next */
        }
      }
      setError('Estudo ainda não disponível. Rode python -m trader study.')
    }
    void load()
  }, [])

  const caseWinners = listWinners(data, bank)
  const winner = caseWinners[Math.min(pick, Math.max(caseWinners.length - 1, 0))]
  const side = lotSide(winner, chartScaled)
  const hourly = useMemo(() => {
    if (!side) return []
    return Object.entries(side.metrics.hourly)
      .map(([hour, v]) => ({ hour: `${hour}h`, ...v }))
      .sort((a, b) => Number(a.hour.replace('h', '')) - Number(b.hour.replace('h', '')))
  }, [side])

  if (error) {
    return (
      <main className="mx-auto max-w-xl px-5 py-24 text-center">
        <AlertTriangle className="mx-auto size-10 text-pink" />
        <p className="mt-4 text-muted-foreground">{error}</p>
      </main>
    )
  }
  if (!data) {
    return (
      <main className="grid min-h-dvh place-items-center">
        <p className="animate-fade-in text-muted-foreground">Carregando o estudo…</p>
      </main>
    )
  }

  const lookback = data.lookback ?? { m1: 15, m5: 3 }
  const leak = data.timeframes.m5?.leakage ?? data.timeframes.m1?.leakage
  const m = side?.metrics
  const periods = side?.by_period ?? winner?.by_period
  const bankNum = Number(bank)
  const scaleHint = `Lote que sobe: 1 mini em R$ 1.000, teto 10 (banca ${brl(bankNum)}).`

  const pickCase = (nextBank?: BankKey) => {
    if (nextBank) setBank(nextBank)
    setPick(0)
    setChartScaled(false)
    document.getElementById('setups')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="relative min-h-dvh">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary/15 via-transparent to-violet/10" />
      <header className="relative mx-auto flex max-w-6xl items-center justify-between px-5 py-6 sm:px-8">
        <p className="font-display text-lg font-bold">Sinal WIN</p>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <a href="/replay">Replay</a>
          </Button>
          <Button variant="outline" size="sm" asChild>
            <a href="/ao-vivo">Ao vivo</a>
          </Button>
          <Button variant="outline" size="sm" onClick={() => document.getElementById('parecer')?.scrollIntoView()}>
            Comparar
          </Button>
          <Button variant="outline" size="sm" onClick={() => document.getElementById('setups')?.scrollIntoView()}>
            Setups
          </Button>
          <Button size="sm" onClick={() => document.getElementById('mt5')?.scrollIntoView()}>
            Até o MT5
          </Button>
        </div>
      </header>

      <section className="relative mx-auto max-w-6xl px-5 pb-12 pt-10 sm:px-8">
        <p className="animate-fade-up text-sm uppercase tracking-[0.2em] text-primary">Estudo para sócios</p>
        <h1 className="mt-4 max-w-3xl animate-fade-up font-display text-4xl font-bold leading-tight sm:text-5xl [animation-delay:80ms]">
          O robô lê os últimos candles de 5 min e sugere compra, venda ou não fazer nada.
        </h1>
        <p className="mt-5 max-w-2xl animate-fade-up text-lg text-muted-foreground [animation-delay:140ms]">
          Mini índice WIN$ contínuo. Treino até dez/2024, teste jan/2025–ago/2026. Sinal no fechamento do M5. Os 15
          candles de 1 min só alimentam essa estratégia. {leak?.n_removed ?? 0} candles repetidos saíram do teste.
        </p>
        <div className="mt-8 grid gap-4 md:grid-cols-1">
          {SIGNAL_CASES.map((item) => (
            <div key={item.key} className="rounded-2xl border border-border bg-elevated/60 p-5">
              <p className="font-display text-lg font-semibold">{item.label}</p>
              <p className="mt-2 text-sm text-muted-foreground">{item.help}</p>
              <p className="mt-3 text-xs text-muted-foreground">
                Contexto: {lookback.m1} × 1 min. Operação: {lookback.m5} × 5 min.
              </p>
            </div>
          ))}
        </div>
        <p className="mt-6 text-sm text-muted-foreground">
          Bancas {BANKS.map((value) => brl(Number(value))).join(', ')}. Tempo gráfico fixo em 5 min. Ponto ={' '}
          {brl(data.instrument.point_value)}.
        </p>
      </section>

      {data.parecer ? (
        <CaseCompare parecer={data.parecer} lookback={lookback} onPick={pickCase} />
      ) : null}

      <section id="setups" className="relative mx-auto max-w-6xl px-5 py-10 sm:px-8">
        <h2 className="font-display text-3xl font-bold">Melhor e segundo melhor</h2>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          Filtre pela banca. Caso e tempo gráfico são fixos: últimos candles em 5 min. {scaleHint} Gerado em{' '}
          {new Date(data.generated_at).toLocaleString('pt-BR')}.
        </p>

        <div className="mt-6 rounded-2xl border border-border bg-elevated/40 p-4">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Banca inicial</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {(data.banks.length ? data.banks : BANKS.map(Number)).map((value) => {
              const key = String(value) as BankKey
              return (
                <Button
                  key={key}
                  variant={bank === key ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {
                    setBank(key)
                    setPick(0)
                    setChartScaled(false)
                  }}
                >
                  {brl(value)}
                </Button>
              )
            })}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{CASE_HELP.last_candles}</p>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {caseWinners.map((w, i) => (
            <WinnerCard
              key={w.params.name || w.name}
              winner={w}
              title={i === 0 ? 'melhor' : 'segundo melhor'}
              active={pick === i}
              onClick={() => setPick(i)}
            />
          ))}
        </div>
        {!winner ? (
          <p className="mt-6 rounded-2xl border border-border bg-elevated/50 p-5 text-sm text-muted-foreground">
            Nenhum setup rodou nesta banca.
          </p>
        ) : null}

        {m && winner && side ? (
          <>
            <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Kpi label="P&L no teste (20 meses)" value={brl(m.net_pnl)} positive={m.net_pnl >= 0} />
              <Kpi label="Maior tombo" value={brl(m.max_drawdown)} hint={pct(m.max_drawdown_pct)} />
              <Kpi label="Acerto" value={pct(m.win_rate)} hint={`${m.n_wins} gains · ${m.n_losses} stops`} />
              <Kpi label="Operações" value={String(m.n_trades)} hint={`fator ${num(m.profit_factor, 2)}`} />
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-5">
              <div className="rounded-2xl border border-border bg-elevated/40 p-4 lg:col-span-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-display font-semibold">Curva da banca</h3>
                  <Button variant="outline" size="sm" onClick={() => setChartScaled((v) => !v)}>
                    {chartScaled ? 'Ver curva em 1 mini' : 'Ver curva do lote que sobe'}
                  </Button>
                </div>
                {chartScaled ? (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Mesmo setup, lote crescente. Tombo pode passar da banca — não entra no ranking.
                  </p>
                ) : null}
                <div className="mt-4 h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={m.equity}>
                      <defs>
                        <linearGradient id="bank" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#058ef2" stopOpacity={0.7} />
                          <stop offset="100%" stopColor="#9f2db3" stopOpacity={0.05} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
                      <XAxis dataKey="t" hide />
                      <YAxis tick={{ fill: '#a1a1aa', fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
                        formatter={(value) => brl(Number(value ?? 0))}
                      />
                      <Area type="monotone" dataKey="bank" stroke="#058ef2" fill="url(#bank)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="rounded-2xl border border-border bg-elevated/40 p-4 lg:col-span-2">
                <h3 className="font-display font-semibold">Por horário</h3>
                <div className="mt-4 h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={hourly}>
                      <CartesianGrid stroke="#3a3a3c" strokeDasharray="3 3" />
                      <XAxis dataKey="hour" tick={{ fill: '#a1a1aa', fontSize: 12 }} />
                      <YAxis tick={{ fill: '#a1a1aa', fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ background: '#1c1c1e', border: '1px solid #3a3a3c', borderRadius: 12 }}
                        formatter={(value) => brl(Number(value ?? 0))}
                      />
                      <Bar dataKey="pnl" radius={[6, 6, 0, 0]}>
                        {hourly.map((row) => (
                          <Cell key={row.hour} fill={row.pnl >= 0 ? '#34d399' : '#fb7185'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {periods ? (
              <div className="mt-8 space-y-4">
                <PeriodBars title="Resultado mensal" rows={periods.monthly} />
                <Button variant="outline" size="sm" onClick={() => setShowIntra((v) => !v)}>
                  {showIntra ? 'Ocultar dias e semanas' : 'Ver dias e semanas'}
                </Button>
                {showIntra ? (
                  <div className="grid gap-4 lg:grid-cols-2">
                    <PeriodBars title="Diário" rows={periods.daily} />
                    <PeriodBars title="Semanal" rows={periods.weekly} />
                  </div>
                ) : null}
              </div>
            ) : null}

            <div className="mt-8">
              <SetupParams winner={winner} />
            </div>
          </>
        ) : null}
      </section>

      <section id="mt5" className="relative mx-auto max-w-6xl px-5 py-10 sm:px-8">
        <h2 className="font-display text-3xl font-bold">Do estudo ao MetaTrader 5</h2>
        <ol className="mt-6 space-y-3">
          {data.mt5.steps.map((step, i) => (
            <li key={step} className="flex gap-4 rounded-2xl border border-border bg-elevated/50 p-4">
              <span className="font-display text-primary">{i + 1}</span>
              <p>{step}</p>
            </li>
          ))}
        </ol>
        {data.insights.improve.length ? (
          <ul className="mt-6 space-y-2 text-sm text-muted-foreground">
            {data.insights.improve.map((item) => (
              <li key={item}>· {item}</li>
            ))}
          </ul>
        ) : null}
        <p className="mt-6 text-sm text-muted-foreground">{data.disclaimer}</p>
        <p className="mt-2 text-xs text-muted-foreground">Gerado em {new Date(data.generated_at).toLocaleString('pt-BR')}</p>
      </section>

      <footer className="border-t border-border py-8 text-center text-sm text-muted-foreground">
        <p>Sinal WIN · estudo educacional · não é recomendação de investimento</p>
        <a
          href="https://koletivo-hub.vercel.app"
          target="_blank"
          rel="noreferrer"
          className="mt-5 inline-flex flex-col items-center gap-3 text-foreground hover:text-primary"
        >
          <img src="/brand/logo-branco.png" alt="Koletivo Hub" className="h-10 w-auto object-contain" />
          <span className="font-display text-sm font-semibold">Desenvolvido por Koletivo Hub</span>
        </a>
        <p className="mt-3 text-xs text-muted-foreground/70">
          © {new Date().getFullYear()}. Todos os direitos reservados a Koletivo Hub.
        </p>
      </footer>
    </div>
  )
}
