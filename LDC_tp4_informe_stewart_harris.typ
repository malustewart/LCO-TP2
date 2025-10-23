#import "@preview/elsearticle:1.0.0": *
#import  "@preview/dashy-todo:0.1.2": todo
#import "@preview/subpar:0.2.2"


#let todo-inline = todo.with(position: "inline", stroke: (paint: red, thickness: 4pt, cap: "round"))

#set page(
  margin:(left: 1cm, right: 1cm)
)

#set math.cases(gap: 0.8em)
#show math.equation.where(block: true): set align(left)

#show: elsearticle.with(
  title: "Laboratorio de comunicaciones\nTP4: Simulación sistema óptico MI/DD",
  authors: (
    (
      name: "María Luz Stewart Harris",
      affiliation: "Instituto Balseiro",
      //corr: "maria.stewart@ib.edu.ar",
    ),
  ),
  numcol: 1,
  // journal: "Name of the Journal",
  // abstract: abstract,
  // keywords: ("keyword 1", "keyword 2"),
  format:"3p",
  // line-numbering: true,
)



= Descripción sistema

Se simula un sistema de comunicación formado por un láser de onda continua, un modulador Mach-Zehnder (mzm), una fibra óptica monomodo estándar, y fotodetector PIN:

#figure(
  image("figs/componentes.png"),
  caption: [Sistema modelado]
)

A menos que se especifique otra cosa, los valores utilizados para los parámetros de las simulaciones son:
#align(center)[
#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Parámetro láser*], [*Valor*],
  ),
  $P_"out"$, "-10dBm",
  $delta nu$, [10 MHz],
  [RIN], [-150dB/Hz],
),

#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Parámetro MZM*], [*Valor*],
  ),
  $V_pi$, [5V],
  $V_"bias"$, [-2.5V],
  [K], [0.8],
  [Er], [30 dB],
),

#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Parámetro PD*], [*Valor*],
  ),
  $R$, [1A/W],
  $B$, [5 GHz],
  $T$, [300 K],
  $R_l$, $50 Omega$,
  $i_d$, [10nA],
  $F_n$, [0 dB],
),

#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Parámetro señal*], [*Valor*],
  ),
  [Modulación], [NRZ],
  $B_r$, [5 Gbps],
)
]
= Impacto de la amplitud de modulación:

#subpar.grid(
  columns: (1fr, 1fr),
  figure(image("figs/ej1_pd_eye_VppOverVpi_0.5.png"), caption: [$V_(p p) = 0.5 V_pi$]), <mzm_0.5>,
  figure(image("figs/ej1_pd_eye_VppOverVpi_0.8.png"), caption: [$V_(p p) = 0.8 V_pi$]), <mzm_0.8>,
  figure(image("figs/ej1_pd_eye_VppOverVpi_1.0.png"), caption: [$V_(p p) = 1.0 V_pi$]), <mzm_1.0>,
  figure(image("figs/ej1_pd_eye_VppOverVpi_1.2.png"), caption: [$V_(p p) = 1.2 V_pi$]), <mzm_1.2>,
  caption: [Comparación de salida del MZM medida por el fotodetector para diferentes amplitudes de modulación.
  ],
  label: <mzm>,
)
#align(center)[
#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*$V_(p p)$ de modulación*], [*Apertura diagrama de ojo*],
  ),
  $0.5 V_(pi)$, $2.1 m V$,
  $0.8 V_(pi)$, $3.1 m V$,
  $1.0 V_(pi)$, $3.5 m V$,
  $1.2 V_(pi)$, $3.1 m V$,
)
]

Asumiendo una $V_"bias"=-0.5V_pi$, la apertura del diagrama de ojo es máxima cuando $V_(p p)= V_pi$. Esto se debe a que la transferencia de potencia en función de la señal de modulación tiene forma cosenoidal ($cos^2$) y este valor de $V_(p p)$ varía la tensión de modulación entre un mínimo y un máximo de la transferencia:

#figure(
  image("figs/ej1_mzm_transfer_function.svg"),
  caption: [Transferencia de potencia del MZM en función de la modulación $u(t)$ ($V_"bias" = -2.5V$).]
)<mzm_trans>

= Impacto de la longitud de la fibra óptica

#subpar.grid(
  columns: (1fr, 1fr),
  figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_40.png"), caption: [$L=40 k m$]), <L_40>,
  figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_60.png"), caption: [$L=60 k m$]), <L_60>,
  figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_80.png"), caption: [$L=80 k m$]), <L_80>,
  figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_100.png"), caption: [$L=100 k m$]), <L_100>,
  caption: [Comparación de salida de la fibra medida por el fotodetector para diferentes longitudes de fibra.\
  $P_"láser" = -10$ dBm, $alpha=0$ dB/km, $beta_2=-20 (p s^2)/(k m)$, $beta_3=0.1 (p s^3)/(k m)$, $gamma=0$
  ],
  label: <L>,
)

#align(center)[
#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Longitud fibra óptica*], [*Apertura diagrama de ojo*],
  ),
  $40 k m$, $2.9 m V$,
  $60 k m$, $2.2 m V$,
  $80 k m$, $2.0 m V$,
  $100 k m$, $0.7 m V$,
)
]

Para ver el efecto de la dispersión cromática en la apertura del diagrama de ojo, la simulación utiliza $alpha=gamma=0$ (es decir, una fibra sin atenuación ni efectos no lineales).
Aumentar la longitud de la fibra reduce la apertura del diagrama de ojo debido al ensanchamiento de pulso causado por la dispersión de segundo orden. El efecto de la dispersión de tercer orden es despreciable:

#nonumeq(
  $
    L_(d 2) = T_0^2/abs(beta_2) = (100 p s)^2/(20 (p s^2) / (k m)) = 500 k m \
    L_(d 3) = T_0^3/abs(beta_3) = (100 p s)^3/(0.1 (p s^3) / (k m)) = 10^7 k m >> L_(d 2)\
  $
)



= Impacto de la potencia del láser

#subpar.grid(
  columns: (1fr, 1fr),
  figure(image("figs/ej3_pd_eye_P_-10_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=-10 "dBm"$]), <P_-10>,
  figure(image("figs/ej3_pd_eye_P_0_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=0"dBm"$]), <P_0>,
  figure(image("figs/ej3_pd_eye_P_10_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=10 "dBm"$]), <P_10>,
  figure(image("figs/ej3_pd_eye_P_20_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=20 "dBm"$]), <P_20>,
  caption: [Comparación de salida de la fibra medida por el fotodetector para diferentes potencias de salida del láser.\
  L=40km, $alpha=0$ dB/km, $beta_2=-20 (p s^2)/(k m)$, $beta_3=0 (p s^3)/(k m)$, $gamma=1.5$],
  label: <P>,
)
#align(center)[
#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*$P_"láser"$*], [*Apertura diagrama de ojo*],
  ),
  $-10 "dBm"$, $3.2 m V$,
  $"   "0 "dBm"$, $34 m V$,
  $" "10 "dBm"$, $350 m V$,
  $" "20 "dBm"$, $1800 m V$,
)
]
Aumentar la potencia del láser aumenta la apertura del diagrama de ojo. Para el aumento desde $-10$dBm hasta $10$ dBm el aumento en la apertura es aproximadamente proporcional al aumento de potencia: aumentar en un factor de 10 la potencia ($Delta P =10$ dBm) aumenta la apertura del diagrama de ojo en un factor de 10. 
#footnote([Para las potencias más bajas, el aumento es levemente mayor porque el piso de ruido era más comparable con la potencia de la señal. Es decir, a potencias más bajas, aumentar la potencia tiene mayor efecto en el aumento de la SNR.])
Esto se debe a que las principales causas de cierre del diagrama de ojo son factores lineales (atenuación y dispersión de segundo orden). 

Al aumentar la potencia de 10 dBm a 20 dBm la apertura del diagrama de ojo no aumenta en un factor de 10 como en los casos anteriores, sino en un factor de aproximadamente 5. Esto se debe a los efectos no lineales que aumentan con el aumento de potencia.

= Impacto tasa de transmisión

#subpar.grid(
  columns: (1fr, 1fr),
  figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_5000000000.0.png"), caption: [$B_r=5 G b p s$]), <Br_5G>,
  figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_10000000000.0.png"), caption: [$B_r=10 G b p s$]), <Br_10G>,
  figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_15000000000.0.png"), caption: [$B_r=15 G b p s$]), <Br_15G>,
  figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_20000000000.0.png"), caption: [$B_r=20 G b p s$]), <Br_20>,
  caption: [Comparación de salida de la fibra medida por el fotodetector para diferentes tasas de transmisión.\
  L=50km, $P_"láser"=10$ dBm, $alpha=0$ dB/km, $beta_2=-20 (p s^2)/(k m)$, $beta_3=0.1 (p s^3)/(k m)$, $gamma=0$
  ],
  label: <Br>,
)

#align(center)[
#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Tasa de transmisión*], [*Apertura diagrama de ojo*],
  ),
  $5 G b p s$, $300 m V$,
  $10 G b p s$, $300 m V$,
  $15 G b p s$, $90 m V$,
  $20 G b p s$, $20 m V$,
)
]
El aumento de la tasa de transmisión reduce la apertura de los diagramas de ojo. Esto se debe a que reducir el ancho del pulso $T_0 = 1/B_r$ aumenta el efecto de la distorsión de segundo orden, caracterizada por $beta_2$. La siguiente tabla muestra como se reduce la longitud de dispersión al aumentar la tasa de transmisión:

#align(center)[
#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Tasa de transmisión*], [*Tiempo de pulso*], [*$L_(d 2)$*],
  ),
  $5 G b p s$, $200 p s$, $2000 k m$,
  $10 G b p s$, $100 p s$, $500 k m$,
  $15 G b p s$, $66.6 p s$, $222 k m$,
  $20 G b p s$, $50 p s$, $125 k m$,
)
]

Notar que por más que la apertura vertical de ojo entre $5 G b p s$ y $10 G b p s$ se mantenga aproximadamente constante, la tasa más alta tiene una apertura horizontal de ojo menor, lo que hace al sistema más sensible al jitter.

= Comparación modulación NRZ y RZ


#subpar.grid(
  columns: (1fr, 1fr, 1fr),
  figure(image("figs/ej5_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_1.5_L_40_mod_RZ.png"), caption: [\
  Modulación RZ\
  $P_"láser"=-10 d B m $\
  $P_"media"=-16 d B m $]), <mod_RZ_-10>,
  figure(image("figs/ej5_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_1.5_L_40_mod_NRZ.png"), caption: [\
  Modulación NRZ\
  $P_"láser"=-10 d B m $\
  $P_"media"=-13 d B m $]), <mod_NRZ_-10>,
  figure(image("figs/ej5_pd_eye_P_-7_a_0_b2_-20_b3_0.1_gamma_1.5_L_40_mod_RZ.png"), caption: [\
  Modulación RZ\
  $P_"láser"=-7 d B m $\
  $P_"media"=-13 d B m $]), <mod_RZ_-7>,
  caption: [Comparación de salida de la fibra medida por el fotodetector para modulación NRZ y RZ.\
  L=40km, $alpha=0$ dB/km, $beta_2=-20 (p s^2)/(k m)$, $beta_3=0.1 (p s^3)/(k m)$, $gamma=1.5$
  ],
  label: <mod>,
)
#align(center)[
#table(
  columns: (auto, auto, auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Modulación*],[*$P_"láser"$*],[*$P_"media"$*],[*Apertura diagrama de ojo*],
  ),
  [RZ],$-10 "dBm"$, $-16"dBm"$, [1.2mV],
  [NRZ],$-10 "dBm"$, $-13"dBm"$, [3.0mV],
  [RZ],$-7 "dBm"$, $-13"dBm"$, [3.0mV],
)
]
Para la misma $P_"láser"$ la modulación NRZ tiene una apertura de diagrama de ojo mayor que la modulación RZ. Esto se debe a que la dispersión de segundo orden tiene menos impacto porque el pulso es el doble de ancho.

Comparando el caso de la misma $P_"media"$, las aperturas verticales del diagrama de ojo son prácticamente equivalentes. Si se centra el umbral de decisión entre el punto mínimo y el máximo de la apertura vertical, la apertura horizontal es levemente mayor en la modulación RZ. 
