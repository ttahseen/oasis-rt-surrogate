зк#
Чж
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceИ
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeКнout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
М
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
∞
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758р±!
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
z
dense_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namedense_output/bias
s
%dense_output/bias/Read/ReadVariableOpReadVariableOpdense_output/bias*
_output_shapes
:*
dtype0
В
dense_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_namedense_output/kernel
{
'dense_output/kernel/Read/ReadVariableOpReadVariableOpdense_output/kernel*
_output_shapes

:@*
dtype0
Ж
gru_1/gru_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*&
shared_namegru_1/gru_cell_3/bias

)gru_1/gru_cell_3/bias/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_3/bias*
_output_shapes

:`*
dtype0
Ю
!gru_1/gru_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*2
shared_name#!gru_1/gru_cell_3/recurrent_kernel
Ч
5gru_1/gru_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_1/gru_cell_3/recurrent_kernel*
_output_shapes

: `*
dtype0
К
gru_1/gru_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*(
shared_namegru_1/gru_cell_3/kernel
Г
+gru_1/gru_cell_3/kernel/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_3/kernel*
_output_shapes

: `*
dtype0
В
gru/gru_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*$
shared_namegru/gru_cell_2/bias
{
'gru/gru_cell_2/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell_2/bias*
_output_shapes

:`*
dtype0
Ъ
gru/gru_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*0
shared_name!gru/gru_cell_2/recurrent_kernel
У
3gru/gru_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell_2/recurrent_kernel*
_output_shapes

: `*
dtype0
Ж
gru/gru_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*&
shared_namegru/gru_cell_2/kernel

)gru/gru_cell_2/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell_2/kernel*
_output_shapes

:`*
dtype0
|
dense_surface/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namedense_surface/bias
u
&dense_surface/bias/Read/ReadVariableOpReadVariableOpdense_surface/bias*
_output_shapes
: *
dtype0
Д
dense_surface/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$ *%
shared_namedense_surface/kernel
}
(dense_surface/kernel/Read/ReadVariableOpReadVariableOpdense_surface/kernel*
_output_shapes

:$ *
dtype0
}
serving_default_inputs_auxPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Ж
serving_default_inputs_mainPlaceholder*+
_output_shapes
:€€€€€€€€€1*
dtype0* 
shape:€€€€€€€€€1
а
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputs_auxserving_default_inputs_maingru/gru_cell_2/biasgru/gru_cell_2/kernelgru/gru_cell_2/recurrent_kerneldense_surface/kerneldense_surface/biasgru_1/gru_cell_3/biasgru_1/gru_cell_3/kernel!gru_1/gru_cell_3/recurrent_kerneldense_output/kerneldense_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_772564

NoOpNoOp
£>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ё=
value‘=B—= B =
ј
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
* 
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
* 

	keras_api* 
¶
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*

'	keras_api* 

(	keras_api* 
Ѕ
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/_random_generator
0cell
1
state_spec*

2	keras_api* 
Ы
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
	9layer*
J
:0
;1
<2
%3
&4
=5
>6
?7
@8
A9*
J
:0
;1
<2
%3
&4
=5
>6
?7
@8
A9*
* 
∞
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 
* 
O
O
_variables
P_iterations
Q_learning_rate
R_update_step_xla*
* 

Sserving_default* 

:0
;1
<2*

:0
;1
<2*
* 
Я

Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ztrace_0
[trace_1
\trace_2
]trace_3* 
6
^trace_0
_trace_1
`trace_2
atrace_3* 
* 
”
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator

:kernel
;recurrent_kernel
<bias*
* 
* 

%0
&1*

%0
&1*
* 
У
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
d^
VARIABLE_VALUEdense_surface/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEdense_surface/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

=0
>1
?2*

=0
>1
?2*
* 
Я

pstates
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
6
vtrace_0
wtrace_1
xtrace_2
ytrace_3* 
6
ztrace_0
{trace_1
|trace_2
}trace_3* 
* 
Ў
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses
Д_random_generator

=kernel
>recurrent_kernel
?bias*
* 
* 

@0
A1*

@0
A1*
* 
Ш
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

Кtrace_0
Лtrace_1* 

Мtrace_0
Нtrace_1* 
ђ
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses

@kernel
Abias*
UO
VARIABLE_VALUEgru/gru_cell_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEgru/gru_cell_2/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEgru/gru_cell_2/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_1/gru_cell_3/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_1/gru_cell_3/recurrent_kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_1/gru_cell_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEdense_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 
J
0
1
2
3
4
5
6
7
	8

9*

Ф0
Х1
Ц2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

P0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

:0
;1
<2*

:0
;1
<2*
* 
Ш
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

Ьtrace_0
Эtrace_1* 

Юtrace_0
Яtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

00*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

=0
>1
?2*

=0
>1
?2*
* 
Ь
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
~	variables
trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses*

•trace_0
¶trace_1* 

Іtrace_0
®trace_1* 
* 
* 

90*
* 
* 
* 
* 
* 
* 
* 

@0
A1*

@0
A1*
* 
Ю
©non_trainable_variables
™layers
Ђmetrics
 ђlayer_regularization_losses
≠layer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses*

Ѓtrace_0* 

ѓtrace_0* 
<
∞	variables
±	keras_api

≤total

≥count*
M
і	variables
µ	keras_api

ґtotal

Јcount
Є
_fn_kwargs*
M
є	variables
Ї	keras_api

їtotal

Љcount
љ
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

≤0
≥1*

∞	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ґ0
Ј1*

і	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

ї0
Љ1*

є	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
о
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_surface/kerneldense_surface/biasgru/gru_cell_2/kernelgru/gru_cell_2/recurrent_kernelgru/gru_cell_2/biasgru_1/gru_cell_3/kernel!gru_1/gru_cell_3/recurrent_kernelgru_1/gru_cell_3/biasdense_output/kerneldense_output/bias	iterationlearning_ratetotal_2count_2total_1count_1totalcountConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_775060
й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_surface/kerneldense_surface/biasgru/gru_cell_2/kernelgru/gru_cell_2/recurrent_kernelgru/gru_cell_2/biasgru_1/gru_cell_3/kernel!gru_1/gru_cell_3/recurrent_kernelgru_1/gru_cell_3/biasdense_output/kerneldense_output/bias	iterationlearning_ratetotal_2count_2total_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_775124ыј 
∞	
ц
gru_while_cond_772680$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1<
8gru_while_gru_while_cond_772680___redundant_placeholder0<
8gru_while_gru_while_cond_772680___redundant_placeholder1<
8gru_while_gru_while_cond_772680___redundant_placeholder2<
8gru_while_gru_while_cond_772680___redundant_placeholder3
gru_while_identity
r
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::N J

_output_shapes
: 
0
_user_specified_namegru/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namegru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
э
’
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_770879

inputs

states)
readvariableop_resource:`0
matmul_readvariableop_resource:`2
 matmul_1_readvariableop_resource: `
identity

identity_1ИҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:€€€€€€€€€ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ [
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
≤

„
+__inference_gru_cell_2_layer_call_fn_774710

inputs
states_0
unknown:`
	unknown_0:`
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_770879o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states_0
Ч

т
A__inference_dense_layer_call_and_return_conditional_losses_771520

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
иM
М
?__inference_gru_layer_call_and_return_conditional_losses_773950

inputs4
"gru_cell_2_readvariableop_resource:`;
)gru_cell_2_matmul_readvariableop_resource:`=
+gru_cell_2_matmul_1_readvariableop_resource: `
identity

identity_1ИҐ gru_cell_2/MatMul/ReadVariableOpҐ"gru_cell_2/MatMul_1/ReadVariableOpҐgru_cell_2/ReadVariableOpҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:1€€€€€€€€€R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask|
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0С
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_773860*
condR
while_cond_773859*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:1€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€1 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€1 _

Identity_1Identitywhile:output:4^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ≤
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€1: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€1
 
_user_specified_nameinputs
э
’
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_771022

inputs

states)
readvariableop_resource:`0
matmul_readvariableop_resource:`2
 matmul_1_readvariableop_resource: `
identity

identity_1ИҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:€€€€€€€€€ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ [
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
¬ 
®
while_body_771035
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_2_771057_0:`+
while_gru_cell_2_771059_0:`+
while_gru_cell_2_771061_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_2_771057:`)
while_gru_cell_2_771059:`)
while_gru_cell_2_771061: `ИҐ(while/gru_cell_2/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0В
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_771057_0while_gru_cell_2_771059_0while_gru_cell_2_771061_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_771022Џ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: О
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w

while/NoOpNoOp)^while/gru_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_2_771057while_gru_cell_2_771057_0"4
while_gru_cell_2_771059while_gru_cell_2_771059_0"4
while_gru_cell_2_771061while_gru_cell_2_771061_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2T
(while/gru_cell_2/StatefulPartitionedCall(while/gru_cell_2/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Љ
™
while_cond_774544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_774544___redundant_placeholder04
0while_while_cond_774544___redundant_placeholder14
0while_while_cond_774544___redundant_placeholder24
0while_while_cond_774544___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
≤

„
+__inference_gru_cell_2_layer_call_fn_774724

inputs
states_0
unknown:`
	unknown_0:`
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_771022o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states_0
њ
М
&__inference_model_layer_call_fn_772358
inputs_main

inputs_aux
unknown:`
	unknown_0:`
	unknown_1: `
	unknown_2:$ 
	unknown_3: 
	unknown_4:`
	unknown_5: `
	unknown_6: `
	unknown_7:@
	unknown_8:
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputs_main
inputs_auxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_772335s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_aux
Й=
щ
while_body_773398
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`C
1while_gru_cell_2_matmul_readvariableop_resource_0:`E
3while_gru_cell_2_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`A
/while_gru_cell_2_matmul_readvariableop_resource:`C
1while_gru_cell_2_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_2/MatMul/ReadVariableOpҐ(while/gru_cell_2/MatMul_1/ReadVariableOpҐwhile/gru_cell_2/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0К
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
КИ
С
__inference__traced_save_775060
file_prefix=
+read_disablecopyonread_dense_surface_kernel:$ 9
+read_1_disablecopyonread_dense_surface_bias: @
.read_2_disablecopyonread_gru_gru_cell_2_kernel:`J
8read_3_disablecopyonread_gru_gru_cell_2_recurrent_kernel: `>
,read_4_disablecopyonread_gru_gru_cell_2_bias:`B
0read_5_disablecopyonread_gru_1_gru_cell_3_kernel: `L
:read_6_disablecopyonread_gru_1_gru_cell_3_recurrent_kernel: `@
.read_7_disablecopyonread_gru_1_gru_cell_3_bias:`>
,read_8_disablecopyonread_dense_output_kernel:@8
*read_9_disablecopyonread_dense_output_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: +
!read_12_disablecopyonread_total_2: +
!read_13_disablecopyonread_count_2: +
!read_14_disablecopyonread_total_1: +
!read_15_disablecopyonread_count_1: )
read_16_disablecopyonread_total: )
read_17_disablecopyonread_count: 
savev2_const
identity_37ИҐMergeV2CheckpointsҐRead/DisableCopyOnReadҐRead/ReadVariableOpҐRead_1/DisableCopyOnReadҐRead_1/ReadVariableOpҐRead_10/DisableCopyOnReadҐRead_10/ReadVariableOpҐRead_11/DisableCopyOnReadҐRead_11/ReadVariableOpҐRead_12/DisableCopyOnReadҐRead_12/ReadVariableOpҐRead_13/DisableCopyOnReadҐRead_13/ReadVariableOpҐRead_14/DisableCopyOnReadҐRead_14/ReadVariableOpҐRead_15/DisableCopyOnReadҐRead_15/ReadVariableOpҐRead_16/DisableCopyOnReadҐRead_16/ReadVariableOpҐRead_17/DisableCopyOnReadҐRead_17/ReadVariableOpҐRead_2/DisableCopyOnReadҐRead_2/ReadVariableOpҐRead_3/DisableCopyOnReadҐRead_3/ReadVariableOpҐRead_4/DisableCopyOnReadҐRead_4/ReadVariableOpҐRead_5/DisableCopyOnReadҐRead_5/ReadVariableOpҐRead_6/DisableCopyOnReadҐRead_6/ReadVariableOpҐRead_7/DisableCopyOnReadҐRead_7/ReadVariableOpҐRead_8/DisableCopyOnReadҐRead_8/ReadVariableOpҐRead_9/DisableCopyOnReadҐRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: }
Read/DisableCopyOnReadDisableCopyOnRead+read_disablecopyonread_dense_surface_kernel"/device:CPU:0*
_output_shapes
 І
Read/ReadVariableOpReadVariableOp+read_disablecopyonread_dense_surface_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:$ *
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:$ a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:$ 
Read_1/DisableCopyOnReadDisableCopyOnRead+read_1_disablecopyonread_dense_surface_bias"/device:CPU:0*
_output_shapes
 І
Read_1/ReadVariableOpReadVariableOp+read_1_disablecopyonread_dense_surface_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: В
Read_2/DisableCopyOnReadDisableCopyOnRead.read_2_disablecopyonread_gru_gru_cell_2_kernel"/device:CPU:0*
_output_shapes
 Ѓ
Read_2/ReadVariableOpReadVariableOp.read_2_disablecopyonread_gru_gru_cell_2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:`М
Read_3/DisableCopyOnReadDisableCopyOnRead8read_3_disablecopyonread_gru_gru_cell_2_recurrent_kernel"/device:CPU:0*
_output_shapes
 Є
Read_3/ReadVariableOpReadVariableOp8read_3_disablecopyonread_gru_gru_cell_2_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: `*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: `c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

: `А
Read_4/DisableCopyOnReadDisableCopyOnRead,read_4_disablecopyonread_gru_gru_cell_2_bias"/device:CPU:0*
_output_shapes
 ђ
Read_4/ReadVariableOpReadVariableOp,read_4_disablecopyonread_gru_gru_cell_2_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:`Д
Read_5/DisableCopyOnReadDisableCopyOnRead0read_5_disablecopyonread_gru_1_gru_cell_3_kernel"/device:CPU:0*
_output_shapes
 ∞
Read_5/ReadVariableOpReadVariableOp0read_5_disablecopyonread_gru_1_gru_cell_3_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: `*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: `e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

: `О
Read_6/DisableCopyOnReadDisableCopyOnRead:read_6_disablecopyonread_gru_1_gru_cell_3_recurrent_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_6/ReadVariableOpReadVariableOp:read_6_disablecopyonread_gru_1_gru_cell_3_recurrent_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: `*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: `e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

: `В
Read_7/DisableCopyOnReadDisableCopyOnRead.read_7_disablecopyonread_gru_1_gru_cell_3_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_7/ReadVariableOpReadVariableOp.read_7_disablecopyonread_gru_1_gru_cell_3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:`А
Read_8/DisableCopyOnReadDisableCopyOnRead,read_8_disablecopyonread_dense_output_kernel"/device:CPU:0*
_output_shapes
 ђ
Read_8/ReadVariableOpReadVariableOp,read_8_disablecopyonread_dense_output_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:@~
Read_9/DisableCopyOnReadDisableCopyOnRead*read_9_disablecopyonread_dense_output_bias"/device:CPU:0*
_output_shapes
 ¶
Read_9/ReadVariableOpReadVariableOp*read_9_disablecopyonread_dense_output_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Э
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 °
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_12/DisableCopyOnReadDisableCopyOnRead!read_12_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 Ы
Read_12/ReadVariableOpReadVariableOp!read_12_disablecopyonread_total_2^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_13/DisableCopyOnReadDisableCopyOnRead!read_13_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 Ы
Read_13/ReadVariableOpReadVariableOp!read_13_disablecopyonread_count_2^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_14/DisableCopyOnReadDisableCopyOnRead!read_14_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_14/ReadVariableOpReadVariableOp!read_14_disablecopyonread_total_1^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_15/DisableCopyOnReadDisableCopyOnRead!read_15_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_15/ReadVariableOpReadVariableOp!read_15_disablecopyonread_count_1^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_16/DisableCopyOnReadDisableCopyOnReadread_16_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_16/ReadVariableOpReadVariableOpread_16_disablecopyonread_total^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_17/DisableCopyOnReadDisableCopyOnReadread_17_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_17/ReadVariableOpReadVariableOpread_17_disablecopyonread_count^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: г
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*М
valueВB€B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHУ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B с
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_36Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_37IdentityIdentity_36:output:0^NoOp*
T0*
_output_shapes
: э
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
Й=
щ
while_body_771648
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`C
1while_gru_cell_2_matmul_readvariableop_resource_0:`E
3while_gru_cell_2_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`A
/while_gru_cell_2_matmul_readvariableop_resource:`C
1while_gru_cell_2_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_2/MatMul/ReadVariableOpҐ(while/gru_cell_2/MatMul_1/ReadVariableOpҐwhile/gru_cell_2/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0К
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
ЂN
ю
A__inference_gru_1_layer_call_and_return_conditional_losses_774634

inputs4
"gru_cell_3_readvariableop_resource:`;
)gru_cell_3_matmul_readvariableop_resource: `=
+gru_cell_3_matmul_1_readvariableop_resource: `
identityИҐ gru_cell_3/MatMul/ReadVariableOpҐ"gru_cell_3/MatMul_1/ReadVariableOpҐgru_cell_3/ReadVariableOpҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: t
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    е
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask|
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource*
_output_shapes

: `*
dtype0С
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_774545*
condR
while_cond_774544*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2 ≤
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2 : : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€2 
 
_user_specified_nameinputs
иM
М
?__inference_gru_layer_call_and_return_conditional_losses_771738

inputs4
"gru_cell_2_readvariableop_resource:`;
)gru_cell_2_matmul_readvariableop_resource:`=
+gru_cell_2_matmul_1_readvariableop_resource: `
identity

identity_1ИҐ gru_cell_2/MatMul/ReadVariableOpҐ"gru_cell_2/MatMul_1/ReadVariableOpҐgru_cell_2/ReadVariableOpҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:1€€€€€€€€€R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask|
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0С
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_771648*
condR
while_cond_771647*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:1€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€1 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€1 _

Identity_1Identitywhile:output:4^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ≤
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€1: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€1
 
_user_specified_nameinputs
∞	
ц
gru_while_cond_773013$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1<
8gru_while_gru_while_cond_773013___redundant_placeholder0<
8gru_while_gru_while_cond_773013___redundant_placeholder1<
8gru_while_gru_while_cond_773013___redundant_placeholder2<
8gru_while_gru_while_cond_773013___redundant_placeholder3
gru_while_identity
r
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::N J

_output_shapes
: 
0
_user_specified_namegru/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namegru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ѕ
Ы
.__inference_dense_surface_layer_call_fn_773959

inputs
unknown:$ 
	unknown_0: 
identityИҐStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_771760o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€$: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
тN
А
A__inference_gru_1_layer_call_and_return_conditional_losses_774324
inputs_04
"gru_cell_3_readvariableop_resource:`;
)gru_cell_3_matmul_readvariableop_resource: `=
+gru_cell_3_matmul_1_readvariableop_resource: `
identityИҐ gru_cell_3/MatMul/ReadVariableOpҐ"gru_cell_3/MatMul_1/ReadVariableOpҐgru_cell_3/ReadVariableOpҐwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    е
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask|
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource*
_output_shapes

: `*
dtype0С
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_774235*
condR
while_cond_774234*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ ≤
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs_0
ы5
ь
A__inference_gru_1_layer_call_and_return_conditional_losses_771446

inputs#
gru_cell_3_771370:`#
gru_cell_3_771372: `#
gru_cell_3_771374: `
identityИҐ"gru_cell_3/StatefulPartitionedCallҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    е
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask«
"gru_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_3_771370gru_cell_3_771372gru_cell_3_771374*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_771369n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_3_771370gru_cell_3_771372gru_cell_3_771374*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_771382*
condR
while_cond_771381*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ s
NoOpNoOp#^gru_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2H
"gru_cell_3/StatefulPartitionedCall"gru_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
≤

„
+__inference_gru_cell_3_layer_call_fn_774816

inputs
states_0
unknown:`
	unknown_0: `
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_771225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states_0
 #
Ћ
A__inference_model_layer_call_and_return_conditional_losses_772401

inputs
inputs_1

gru_772365:`

gru_772367:`

gru_772369: `&
dense_surface_772375:$ "
dense_surface_772377: 
gru_1_772384:`
gru_1_772386: `
gru_1_772388: `%
dense_output_772393:@!
dense_output_772395:
identityИҐ$dense_output/StatefulPartitionedCallҐ%dense_surface/StatefulPartitionedCallҐgru/StatefulPartitionedCallҐgru_1/StatefulPartitionedCallЕ
gru/StatefulPartitionedCallStatefulPartitionedCallinputs
gru_772365
gru_772367
gru_772369*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:€€€€€€€€€1 :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_772099Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ђ
tf.concat_3/concatConcatV2$gru/StatefulPartitionedCall:output:1inputs_1 tf.concat_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€$Ь
%dense_surface/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_surface_772375dense_surface_772377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_771760o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€       ™
tf.reshape_1/ReshapeReshape.dense_surface/StatefulPartitionedCall:output:0#tf.reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ƒ
tf.concat_4/concatConcatV2$gru/StatefulPartitionedCall:output:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2 Р
gru_1/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0gru_1_772384gru_1_772386gru_1_772388*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_772273Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ƒ
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0&gru_1/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2@Ь
$dense_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_output_772393dense_output_772395*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_771551k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   У
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@А
IdentityIdentity-dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2”
NoOpNoOp%^dense_output/StatefulPartitionedCall&^dense_surface/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall2N
%dense_surface/StatefulPartitionedCall%dense_surface/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€1
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Й=
щ
while_body_773706
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`C
1while_gru_cell_2_matmul_readvariableop_resource_0:`E
3while_gru_cell_2_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`A
/while_gru_cell_2_matmul_readvariableop_resource:`C
1while_gru_cell_2_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_2/MatMul/ReadVariableOpҐ(while/gru_cell_2/MatMul_1/ReadVariableOpҐwhile/gru_cell_2/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0К
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Љ
™
while_cond_774389
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_774389___redundant_placeholder04
0while_while_cond_774389___redundant_placeholder14
0while_while_cond_774389___redundant_placeholder24
0while_while_cond_774389___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
†

ъ
I__inference_dense_surface_layer_call_and_return_conditional_losses_771760

inputs0
matmul_readvariableop_resource:$ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
њ
М
&__inference_model_layer_call_fn_772424
inputs_main

inputs_aux
unknown:`
	unknown_0:`
	unknown_1: `
	unknown_2:$ 
	unknown_3: 
	unknown_4:`
	unknown_5: `
	unknown_6: `
	unknown_7:@
	unknown_8:
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputs_main
inputs_auxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_772401s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_aux
ЂN
ю
A__inference_gru_1_layer_call_and_return_conditional_losses_774479

inputs4
"gru_cell_3_readvariableop_resource:`;
)gru_cell_3_matmul_readvariableop_resource: `=
+gru_cell_3_matmul_1_readvariableop_resource: `
identityИҐ gru_cell_3/MatMul/ReadVariableOpҐ"gru_cell_3/MatMul_1/ReadVariableOpҐgru_cell_3/ReadVariableOpҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: t
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    е
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask|
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource*
_output_shapes

: `*
dtype0С
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_774390*
condR
while_cond_774389*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2 ≤
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2 : : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€2 
 
_user_specified_nameinputs
Љ
™
while_cond_770891
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_770891___redundant_placeholder04
0while_while_cond_770891___redundant_placeholder14
0while_while_cond_770891___redundant_placeholder24
0while_while_cond_770891___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Й=
щ
while_body_771835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_3_readvariableop_resource_0:`C
1while_gru_cell_3_matmul_readvariableop_resource_0: `E
3while_gru_cell_3_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_3_readvariableop_resource:`A
/while_gru_cell_3_matmul_readvariableop_resource: `C
1while_gru_cell_3_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_3/MatMul/ReadVariableOpҐ(while/gru_cell_3/MatMul_1/ReadVariableOpҐwhile/gru_cell_3/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0К
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: `*
dtype0µ
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_3/MatMul/ReadVariableOp)^while/gru_cell_3/MatMul_1/ReadVariableOp ^while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_3_matmul_1_readvariableop_resource3while_gru_cell_3_matmul_1_readvariableop_resource_0"d
/while_gru_cell_3_matmul_readvariableop_resource1while_gru_cell_3_matmul_readvariableop_resource_0"V
(while_gru_cell_3_readvariableop_resource*while_gru_cell_3_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_3/MatMul/ReadVariableOp&while/gru_cell_3/MatMul/ReadVariableOp2T
(while/gru_cell_3/MatMul_1/ReadVariableOp(while/gru_cell_3/MatMul_1/ReadVariableOp2B
while/gru_cell_3/ReadVariableOpwhile/gru_cell_3/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
∞
З
&__inference_model_layer_call_fn_772590
inputs_0
inputs_1
unknown:`
	unknown_0:`
	unknown_1: `
	unknown_2:$ 
	unknown_3: 
	unknown_4:`
	unknown_5: `
	unknown_6: `
	unknown_7:@
	unknown_8:
identityИҐStatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_772335s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€1
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1
Ы
√
H__inference_dense_output_layer_call_and_return_conditional_losses_771531

inputs
dense_771521:@
dense_771523:
identityИҐdense/StatefulPartitionedCallI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@с
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_771521dense_771523*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_771520\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Х
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€@: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Б
Ъ
-__inference_dense_output_layer_call_fn_774643

inputs
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_771531|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
©
ґ
&__inference_gru_1_layer_call_fn_773981
inputs_0
unknown:`
	unknown_0: `
	unknown_1: `
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_771302|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs_0
µ	
¬
$__inference_gru_layer_call_fn_773321

inputs
unknown:`
	unknown_0:`
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:€€€€€€€€€1 :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_771738s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€1 q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€1
 
_user_specified_nameinputs
Љ
™
while_cond_773551
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_773551___redundant_placeholder04
0while_while_cond_773551___redundant_placeholder14
0while_while_cond_773551___redundant_placeholder24
0while_while_cond_773551___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Љ
™
while_cond_772183
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_772183___redundant_placeholder04
0while_while_cond_772183___redundant_placeholder14
0while_while_cond_772183___redundant_placeholder24
0while_while_cond_772183___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
њ
У
&__inference_dense_layer_call_fn_774917

inputs
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_771520o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
иM
М
?__inference_gru_layer_call_and_return_conditional_losses_772099

inputs4
"gru_cell_2_readvariableop_resource:`;
)gru_cell_2_matmul_readvariableop_resource:`=
+gru_cell_2_matmul_1_readvariableop_resource: `
identity

identity_1ИҐ gru_cell_2/MatMul/ReadVariableOpҐ"gru_cell_2/MatMul_1/ReadVariableOpҐgru_cell_2/ReadVariableOpҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:1€€€€€€€€€R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask|
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0С
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_772009*
condR
while_cond_772008*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:1€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€1 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€1 _

Identity_1Identitywhile:output:4^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ≤
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€1: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€1
 
_user_specified_nameinputs
Љ
™
while_cond_771381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_771381___redundant_placeholder04
0while_while_cond_771381___redundant_placeholder14
0while_while_cond_771381___redundant_placeholder24
0while_while_cond_771381___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Љ
™
while_cond_774079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_774079___redundant_placeholder04
0while_while_cond_774079___redundant_placeholder14
0while_while_cond_774079___redundant_placeholder24
0while_while_cond_774079___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Љ
™
while_cond_773859
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_773859___redundant_placeholder04
0while_while_cond_773859___redundant_placeholder14
0while_while_cond_773859___redundant_placeholder24
0while_while_cond_773859___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
э
’
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_771225

inputs

states)
readvariableop_resource:`0
matmul_readvariableop_resource: `2
 matmul_1_readvariableop_resource: `
identity

identity_1ИҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: `*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:€€€€€€€€€ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ [
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
Љ
™
while_cond_772008
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_772008___redundant_placeholder04
0while_while_cond_772008___redundant_placeholder14
0while_while_cond_772008___redundant_placeholder24
0while_while_cond_772008___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
 #
Ћ
A__inference_model_layer_call_and_return_conditional_losses_772335

inputs
inputs_1

gru_772299:`

gru_772301:`

gru_772303: `&
dense_surface_772309:$ "
dense_surface_772311: 
gru_1_772318:`
gru_1_772320: `
gru_1_772322: `%
dense_output_772327:@!
dense_output_772329:
identityИҐ$dense_output/StatefulPartitionedCallҐ%dense_surface/StatefulPartitionedCallҐgru/StatefulPartitionedCallҐgru_1/StatefulPartitionedCallЕ
gru/StatefulPartitionedCallStatefulPartitionedCallinputs
gru_772299
gru_772301
gru_772303*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:€€€€€€€€€1 :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_771738Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ђ
tf.concat_3/concatConcatV2$gru/StatefulPartitionedCall:output:1inputs_1 tf.concat_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€$Ь
%dense_surface/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_surface_772309dense_surface_772311*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_771760o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€       ™
tf.reshape_1/ReshapeReshape.dense_surface/StatefulPartitionedCall:output:0#tf.reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ƒ
tf.concat_4/concatConcatV2$gru/StatefulPartitionedCall:output:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2 Р
gru_1/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0gru_1_772318gru_1_772320gru_1_772322*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_771924Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ƒ
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0&gru_1/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2@Ь
$dense_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_output_772327dense_output_772329*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_771531k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   У
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@А
IdentityIdentity-dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2”
NoOpNoOp%^dense_output/StatefulPartitionedCall&^dense_surface/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall2N
%dense_surface/StatefulPartitionedCall%dense_surface/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€1
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ы5
ь
A__inference_gru_1_layer_call_and_return_conditional_losses_771302

inputs#
gru_cell_3_771226:`#
gru_cell_3_771228: `#
gru_cell_3_771230: `
identityИҐ"gru_cell_3/StatefulPartitionedCallҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    е
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask«
"gru_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_3_771226gru_cell_3_771228gru_cell_3_771230*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_771225n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_3_771226gru_cell_3_771228gru_cell_3_771230*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_771238*
condR
while_cond_771237*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ s
NoOpNoOp#^gru_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2H
"gru_cell_3/StatefulPartitionedCall"gru_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ЂN
ю
A__inference_gru_1_layer_call_and_return_conditional_losses_771924

inputs4
"gru_cell_3_readvariableop_resource:`;
)gru_cell_3_matmul_readvariableop_resource: `=
+gru_cell_3_matmul_1_readvariableop_resource: `
identityИҐ gru_cell_3/MatMul/ReadVariableOpҐ"gru_cell_3/MatMul_1/ReadVariableOpҐgru_cell_3/ReadVariableOpҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: t
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    е
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask|
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource*
_output_shapes

: `*
dtype0С
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_771835*
condR
while_cond_771834*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2 ≤
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2 : : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€2 
 
_user_specified_nameinputs
€
і
&__inference_gru_1_layer_call_fn_774003

inputs
unknown:`
	unknown_0: `
	unknown_1: `
identityИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_771924s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€2 
 
_user_specified_nameinputs
Љ
™
while_cond_771237
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_771237___redundant_placeholder04
0while_while_cond_771237___redundant_placeholder14
0while_while_cond_771237___redundant_placeholder24
0while_while_cond_771237___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Й=
щ
while_body_774545
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_3_readvariableop_resource_0:`C
1while_gru_cell_3_matmul_readvariableop_resource_0: `E
3while_gru_cell_3_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_3_readvariableop_resource:`A
/while_gru_cell_3_matmul_readvariableop_resource: `C
1while_gru_cell_3_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_3/MatMul/ReadVariableOpҐ(while/gru_cell_3/MatMul_1/ReadVariableOpҐwhile/gru_cell_3/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0К
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: `*
dtype0µ
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_3/MatMul/ReadVariableOp)^while/gru_cell_3/MatMul_1/ReadVariableOp ^while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_3_matmul_1_readvariableop_resource3while_gru_cell_3_matmul_1_readvariableop_resource_0"d
/while_gru_cell_3_matmul_readvariableop_resource1while_gru_cell_3_matmul_readvariableop_resource_0"V
(while_gru_cell_3_readvariableop_resource*while_gru_cell_3_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_3/MatMul/ReadVariableOp&while/gru_cell_3/MatMul/ReadVariableOp2T
(while/gru_cell_3/MatMul_1/ReadVariableOp(while/gru_cell_3/MatMul_1/ReadVariableOp2B
while/gru_cell_3/ReadVariableOpwhile/gru_cell_3/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Ы
√
H__inference_dense_output_layer_call_and_return_conditional_losses_771551

inputs
dense_771541:@
dense_771543:
identityИҐdense/StatefulPartitionedCallI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@с
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_771541dense_771543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_771520\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Х
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€@: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
б#
“
A__inference_model_layer_call_and_return_conditional_losses_771942
inputs_main

inputs_aux

gru_771739:`

gru_771741:`

gru_771743: `&
dense_surface_771761:$ "
dense_surface_771763: 
gru_1_771925:`
gru_1_771927: `
gru_1_771929: `%
dense_output_771934:@!
dense_output_771936:
identityИҐ$dense_output/StatefulPartitionedCallҐ%dense_surface/StatefulPartitionedCallҐgru/StatefulPartitionedCallҐgru_1/StatefulPartitionedCallК
gru/StatefulPartitionedCallStatefulPartitionedCallinputs_main
gru_771739
gru_771741
gru_771743*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:€€€€€€€€€1 :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_771738Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :≠
tf.concat_3/concatConcatV2$gru/StatefulPartitionedCall:output:1
inputs_aux tf.concat_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€$Ь
%dense_surface/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_surface_771761dense_surface_771763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_771760o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€       ™
tf.reshape_1/ReshapeReshape.dense_surface/StatefulPartitionedCall:output:0#tf.reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ƒ
tf.concat_4/concatConcatV2$gru/StatefulPartitionedCall:output:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2 Р
gru_1/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0gru_1_771925gru_1_771927gru_1_771929*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_771924Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ƒ
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0&gru_1/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2@Ь
$dense_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_output_771934dense_output_771936*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_771531k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   У
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@А
IdentityIdentity-dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2”
NoOpNoOp%^dense_output/StatefulPartitionedCall&^dense_surface/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall2N
%dense_surface/StatefulPartitionedCall%dense_surface/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_aux
Е
„
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_774869

inputs
states_0)
readvariableop_resource:`0
matmul_readvariableop_resource: `2
 matmul_1_readvariableop_resource: `
identity

identity_1ИҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: `*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:€€€€€€€€€ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ [
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states_0
ЂN
ю
A__inference_gru_1_layer_call_and_return_conditional_losses_772273

inputs4
"gru_cell_3_readvariableop_resource:`;
)gru_cell_3_matmul_readvariableop_resource: `=
+gru_cell_3_matmul_1_readvariableop_resource: `
identityИҐ gru_cell_3/MatMul/ReadVariableOpҐ"gru_cell_3/MatMul_1/ReadVariableOpҐgru_cell_3/ReadVariableOpҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: t
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    е
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask|
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource*
_output_shapes

: `*
dtype0С
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_772184*
condR
while_cond_772183*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2 ≤
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2 : : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€2 
 
_user_specified_nameinputs
ТB
с
gru_while_body_773014$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0@
.gru_while_gru_cell_2_readvariableop_resource_0:`G
5gru_while_gru_cell_2_matmul_readvariableop_resource_0:`I
7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0: `
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor>
,gru_while_gru_cell_2_readvariableop_resource:`E
3gru_while_gru_cell_2_matmul_readvariableop_resource:`G
5gru_while_gru_cell_2_matmul_1_readvariableop_resource: `ИҐ*gru/while/gru_cell_2/MatMul/ReadVariableOpҐ,gru/while/gru_cell_2/MatMul_1/ReadVariableOpҐ#gru/while/gru_cell_2/ReadVariableOpМ
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ї
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Т
#gru/while/gru_cell_2/ReadVariableOpReadVariableOp.gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype0Й
gru/while/gru_cell_2/unstackUnpack+gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num†
*gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp5gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype0Ѕ
gru/while/gru_cell_2/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:02gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`І
gru/while/gru_cell_2/BiasAddBiasAdd%gru/while/gru_cell_2/MatMul:product:0%gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`o
$gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€в
gru/while/gru_cell_2/splitSplit-gru/while/gru_cell_2/split/split_dim:output:0%gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split§
,gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0®
gru/while/gru_cell_2/MatMul_1MatMulgru_while_placeholder_24gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ђ
gru/while/gru_cell_2/BiasAdd_1BiasAdd'gru/while/gru_cell_2/MatMul_1:product:0%gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`o
gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€q
&gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ъ
gru/while/gru_cell_2/split_1SplitV'gru/while/gru_cell_2/BiasAdd_1:output:0#gru/while/gru_cell_2/Const:output:0/gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЯ
gru/while/gru_cell_2/addAddV2#gru/while/gru_cell_2/split:output:0%gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru/while/gru_cell_2/SigmoidSigmoidgru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ °
gru/while/gru_cell_2/add_1AddV2#gru/while/gru_cell_2/split:output:1%gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ {
gru/while/gru_cell_2/Sigmoid_1Sigmoidgru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
gru/while/gru_cell_2/mulMul"gru/while/gru_cell_2/Sigmoid_1:y:0%gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Ш
gru/while/gru_cell_2/add_2AddV2#gru/while/gru_cell_2/split:output:2gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ s
gru/while/gru_cell_2/ReluRelugru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ О
gru/while/gru_cell_2/mul_1Mul gru/while/gru_cell_2/Sigmoid:y:0gru_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ _
gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
gru/while/gru_cell_2/subSub#gru/while/gru_cell_2/sub/x:output:0 gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
gru/while/gru_cell_2/mul_2Mulgru/while/gru_cell_2/sub:z:0'gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
gru/while/gru_cell_2/add_3AddV2gru/while/gru_cell_2/mul_1:z:0gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“Q
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: Т
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: Г
gru/while/Identity_4Identitygru/while/gru_cell_2/add_3:z:0^gru/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ “
gru/while/NoOpNoOp+^gru/while/gru_cell_2/MatMul/ReadVariableOp-^gru/while/gru_cell_2/MatMul_1/ReadVariableOp$^gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "p
5gru_while_gru_cell_2_matmul_1_readvariableop_resource7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"l
3gru_while_gru_cell_2_matmul_readvariableop_resource5gru_while_gru_cell_2_matmul_readvariableop_resource_0"^
,gru_while_gru_cell_2_readvariableop_resource.gru_while_gru_cell_2_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"Є
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2X
*gru/while/gru_cell_2/MatMul/ReadVariableOp*gru/while/gru_cell_2/MatMul/ReadVariableOp2\
,gru/while/gru_cell_2/MatMul_1/ReadVariableOp,gru/while/gru_cell_2/MatMul_1/ReadVariableOp2J
#gru/while/gru_cell_2/ReadVariableOp#gru/while/gru_cell_2/ReadVariableOp:N J

_output_shapes
: 
0
_user_specified_namegru/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namegru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
∞
З
&__inference_model_layer_call_fn_772616
inputs_0
inputs_1
unknown:`
	unknown_0:`
	unknown_1: `
	unknown_2:$ 
	unknown_3: 
	unknown_4:`
	unknown_5: `
	unknown_6: `
	unknown_7:@
	unknown_8:
identityИҐStatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_772401s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€1
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1
¬ 
®
while_body_771238
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_3_771260_0:`+
while_gru_cell_3_771262_0: `+
while_gru_cell_3_771264_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_3_771260:`)
while_gru_cell_3_771262: `)
while_gru_cell_3_771264: `ИҐ(while/gru_cell_3/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0В
(while/gru_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_3_771260_0while_gru_cell_3_771262_0while_gru_cell_3_771264_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_771225Џ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: О
while/Identity_4Identity1while/gru_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w

while/NoOpNoOp)^while/gru_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_3_771260while_gru_cell_3_771260_0"4
while_gru_cell_3_771262while_gru_cell_3_771262_0"4
while_gru_cell_3_771264while_gru_cell_3_771264_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2T
(while/gru_cell_3/StatefulPartitionedCall(while/gru_cell_3/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
шM
я

"__inference__traced_restore_775124
file_prefix7
%assignvariableop_dense_surface_kernel:$ 3
%assignvariableop_1_dense_surface_bias: :
(assignvariableop_2_gru_gru_cell_2_kernel:`D
2assignvariableop_3_gru_gru_cell_2_recurrent_kernel: `8
&assignvariableop_4_gru_gru_cell_2_bias:`<
*assignvariableop_5_gru_1_gru_cell_3_kernel: `F
4assignvariableop_6_gru_1_gru_cell_3_recurrent_kernel: `:
(assignvariableop_7_gru_1_gru_cell_3_bias:`8
&assignvariableop_8_dense_output_kernel:@2
$assignvariableop_9_dense_output_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: %
assignvariableop_12_total_2: %
assignvariableop_13_count_2: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*М
valueВB€B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B э
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOpAssignVariableOp%assignvariableop_dense_surface_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_1AssignVariableOp%assignvariableop_1_dense_surface_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_2AssignVariableOp(assignvariableop_2_gru_gru_cell_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_3AssignVariableOp2assignvariableop_3_gru_gru_cell_2_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_4AssignVariableOp&assignvariableop_4_gru_gru_cell_2_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_5AssignVariableOp*assignvariableop_5_gru_1_gru_cell_3_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_6AssignVariableOp4assignvariableop_6_gru_1_gru_cell_3_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_7AssignVariableOp(assignvariableop_7_gru_1_gru_cell_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_8AssignVariableOp&assignvariableop_8_dense_output_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_9AssignVariableOp$assignvariableop_9_dense_output_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_2Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_2Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 џ
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: »
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Й=
щ
while_body_774390
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_3_readvariableop_resource_0:`C
1while_gru_cell_3_matmul_readvariableop_resource_0: `E
3while_gru_cell_3_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_3_readvariableop_resource:`A
/while_gru_cell_3_matmul_readvariableop_resource: `C
1while_gru_cell_3_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_3/MatMul/ReadVariableOpҐ(while/gru_cell_3/MatMul_1/ReadVariableOpҐwhile/gru_cell_3/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0К
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: `*
dtype0µ
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_3/MatMul/ReadVariableOp)^while/gru_cell_3/MatMul_1/ReadVariableOp ^while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_3_matmul_1_readvariableop_resource3while_gru_cell_3_matmul_1_readvariableop_resource_0"d
/while_gru_cell_3_matmul_readvariableop_resource1while_gru_cell_3_matmul_readvariableop_resource_0"V
(while_gru_cell_3_readvariableop_resource*while_gru_cell_3_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_3/MatMul/ReadVariableOp&while/gru_cell_3/MatMul/ReadVariableOp2T
(while/gru_cell_3/MatMul_1/ReadVariableOp(while/gru_cell_3/MatMul_1/ReadVariableOp2B
while/gru_cell_3/ReadVariableOpwhile/gru_cell_3/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Й=
щ
while_body_773552
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`C
1while_gru_cell_2_matmul_readvariableop_resource_0:`E
3while_gru_cell_2_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`A
/while_gru_cell_2_matmul_readvariableop_resource:`C
1while_gru_cell_2_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_2/MatMul/ReadVariableOpҐ(while/gru_cell_2/MatMul_1/ReadVariableOpҐwhile/gru_cell_2/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0К
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
¶N
О
?__inference_gru_layer_call_and_return_conditional_losses_773488
inputs_04
"gru_cell_2_readvariableop_resource:`;
)gru_cell_2_matmul_readvariableop_resource:`=
+gru_cell_2_matmul_1_readvariableop_resource: `
identity

identity_1ИҐ gru_cell_2/MatMul/ReadVariableOpҐ"gru_cell_2/MatMul_1/ReadVariableOpҐgru_cell_2/ReadVariableOpҐwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask|
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0С
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_773398*
condR
while_cond_773397*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ _

Identity_1Identitywhile:output:4^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ≤
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
Љ
™
while_cond_771834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_771834___redundant_placeholder04
0while_while_cond_771834___redundant_placeholder14
0while_while_cond_771834___redundant_placeholder24
0while_while_cond_771834___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ђ«
Ъ	
A__inference_model_layer_call_and_return_conditional_losses_773282
inputs_0
inputs_18
&gru_gru_cell_2_readvariableop_resource:`?
-gru_gru_cell_2_matmul_readvariableop_resource:`A
/gru_gru_cell_2_matmul_1_readvariableop_resource: `>
,dense_surface_matmul_readvariableop_resource:$ ;
-dense_surface_biasadd_readvariableop_resource: :
(gru_1_gru_cell_3_readvariableop_resource:`A
/gru_1_gru_cell_3_matmul_readvariableop_resource: `C
1gru_1_gru_cell_3_matmul_1_readvariableop_resource: `C
1dense_output_dense_matmul_readvariableop_resource:@@
2dense_output_dense_biasadd_readvariableop_resource:
identityИҐ)dense_output/dense/BiasAdd/ReadVariableOpҐ(dense_output/dense/MatMul/ReadVariableOpҐ$dense_surface/BiasAdd/ReadVariableOpҐ#dense_surface/MatMul/ReadVariableOpҐ$gru/gru_cell_2/MatMul/ReadVariableOpҐ&gru/gru_cell_2/MatMul_1/ReadVariableOpҐgru/gru_cell_2/ReadVariableOpҐ	gru/whileҐ&gru_1/gru_cell_3/MatMul/ReadVariableOpҐ(gru_1/gru_cell_3/MatMul_1/ReadVariableOpҐgru_1/gru_cell_3/ReadVariableOpҐgru_1/whileO
	gru/ShapeShapeinputs_0*
T0*
_output_shapes
::нѕa
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
gru/transpose	Transposeinputs_0gru/transpose/perm:output:0*
T0*+
_output_shapes
:1€€€€€€€€€Z
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
::нѕc
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ј
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“К
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“c
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:э
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskД
gru/gru_cell_2/ReadVariableOpReadVariableOp&gru_gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype0}
gru/gru_cell_2/unstackUnpack%gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numТ
$gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp-gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0Э
gru/gru_cell_2/MatMulMatMulgru/strided_slice_2:output:0,gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Х
gru/gru_cell_2/BiasAddBiasAddgru/gru_cell_2/MatMul:product:0gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`i
gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€–
gru/gru_cell_2/splitSplit'gru/gru_cell_2/split/split_dim:output:0gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЦ
&gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp/gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Ч
gru/gru_cell_2/MatMul_1MatMulgru/zeros:output:0.gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Щ
gru/gru_cell_2/BiasAdd_1BiasAdd!gru/gru_cell_2/MatMul_1:product:0gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`i
gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€k
 gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€В
gru/gru_cell_2/split_1SplitV!gru/gru_cell_2/BiasAdd_1:output:0gru/gru_cell_2/Const:output:0)gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitН
gru/gru_cell_2/addAddV2gru/gru_cell_2/split:output:0gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ k
gru/gru_cell_2/SigmoidSigmoidgru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ П
gru/gru_cell_2/add_1AddV2gru/gru_cell_2/split:output:1gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ o
gru/gru_cell_2/Sigmoid_1Sigmoidgru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ К
gru/gru_cell_2/mulMulgru/gru_cell_2/Sigmoid_1:y:0gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Ж
gru/gru_cell_2/add_2AddV2gru/gru_cell_2/split:output:2gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ g
gru/gru_cell_2/ReluRelugru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ }
gru/gru_cell_2/mul_1Mulgru/gru_cell_2/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ж
gru/gru_cell_2/subSubgru/gru_cell_2/sub/x:output:0gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ И
gru/gru_cell_2/mul_2Mulgru/gru_cell_2/sub:z:0!gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru/gru_cell_2/add_3AddV2gru/gru_cell_2/mul_1:z:0gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ƒ
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“J
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€X
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : п
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0&gru_gru_cell_2_readvariableop_resource-gru_gru_cell_2_matmul_readvariableop_resource/gru_gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *!
bodyR
gru_while_body_773014*!
condR
gru_while_cond_773013*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Е
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ќ
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:1€€€€€€€€€ *
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€e
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ы
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ґ
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€1 _
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Щ
tf.concat_3/concatConcatV2gru/while:output:4inputs_1 tf.concat_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€$Р
#dense_surface/MatMul/ReadVariableOpReadVariableOp,dense_surface_matmul_readvariableop_resource*
_output_shapes

:$ *
dtype0Ъ
dense_surface/MatMulMatMultf.concat_3/concat:output:0+dense_surface/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ О
$dense_surface/BiasAdd/ReadVariableOpReadVariableOp-dense_surface_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0†
dense_surface/BiasAddBiasAdddense_surface/MatMul:product:0,dense_surface/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ l
dense_surface/ReluReludense_surface/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€       Ь
tf.reshape_1/ReshapeReshape dense_surface/Relu:activations:0#tf.reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :≥
tf.concat_4/concatConcatV2gru/transpose_1:y:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2 d
gru_1/ShapeShapetf.concat_4/concat:output:0*
T0*
_output_shapes
::нѕc
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Е
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          О
gru_1/transpose	Transposetf.concat_4/concat:output:0gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ ^
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
::нѕe
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“^
gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ж
gru_1/ReverseV2	ReverseV2gru_1/transpose:y:0gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ М
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ч
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/ReverseV2:output:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“e
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskИ
gru_1/gru_cell_3/ReadVariableOpReadVariableOp(gru_1_gru_cell_3_readvariableop_resource*
_output_shapes

:`*
dtype0Б
gru_1/gru_cell_3/unstackUnpack'gru_1/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numЦ
&gru_1/gru_cell_3/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_3_matmul_readvariableop_resource*
_output_shapes

: `*
dtype0£
gru_1/gru_cell_3/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
gru_1/gru_cell_3/BiasAddBiasAdd!gru_1/gru_cell_3/MatMul:product:0!gru_1/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 gru_1/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
gru_1/gru_cell_3/splitSplit)gru_1/gru_cell_3/split/split_dim:output:0!gru_1/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЪ
(gru_1/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_3_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Э
gru_1/gru_cell_3/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
gru_1/gru_cell_3/BiasAdd_1BiasAdd#gru_1/gru_cell_3/MatMul_1:product:0!gru_1/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
gru_1/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"gru_1/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
gru_1/gru_cell_3/split_1SplitV#gru_1/gru_cell_3/BiasAdd_1:output:0gru_1/gru_cell_3/Const:output:0+gru_1/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
gru_1/gru_cell_3/addAddV2gru_1/gru_cell_3/split:output:0!gru_1/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
gru_1/gru_cell_3/SigmoidSigmoidgru_1/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
gru_1/gru_cell_3/add_1AddV2gru_1/gru_cell_3/split:output:1!gru_1/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
gru_1/gru_cell_3/Sigmoid_1Sigmoidgru_1/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
gru_1/gru_cell_3/mulMulgru_1/gru_cell_3/Sigmoid_1:y:0!gru_1/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
gru_1/gru_cell_3/add_2AddV2gru_1/gru_cell_3/split:output:2gru_1/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
gru_1/gru_cell_3/ReluRelugru_1/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_1/gru_cell_3/mul_1Mulgru_1/gru_cell_3/Sigmoid:y:0gru_1/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ [
gru_1/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
gru_1/gru_cell_3/subSubgru_1/gru_cell_3/sub/x:output:0gru_1/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
gru_1/gru_cell_3/mul_2Mulgru_1/gru_cell_3/sub:z:0#gru_1/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
gru_1/gru_cell_3/add_3AddV2gru_1/gru_cell_3/mul_1:z:0gru_1/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€     
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“L

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_3_readvariableop_resource/gru_1_gru_cell_3_matmul_readvariableop_resource1gru_1_gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_1_while_body_773178*#
condR
gru_1_while_cond_773177*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations З
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ‘
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€ *
element_dtype0n
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€g
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:•
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 a
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :≥
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0gru_1/transpose_1:y:0 tf.concat_5/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2@k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   У
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
(dense_output/dense/MatMul/ReadVariableOpReadVariableOp1dense_output_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¶
dense_output/dense/MatMulMatMuldense_output/Reshape:output:00dense_output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)dense_output/dense/BiasAdd/ReadVariableOpReadVariableOp2dense_output_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
dense_output/dense/BiasAddBiasAdd#dense_output/dense/MatMul:product:01dense_output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
dense_output/dense/SigmoidSigmoid#dense_output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
dense_output/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€2      Ю
dense_output/Reshape_1Reshapedense_output/dense/Sigmoid:y:0%dense_output/Reshape_1/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2m
dense_output/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Ч
dense_output/Reshape_2Reshapetf.concat_5/concat:output:0%dense_output/Reshape_2/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@r
IdentityIdentitydense_output/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2к
NoOpNoOp*^dense_output/dense/BiasAdd/ReadVariableOp)^dense_output/dense/MatMul/ReadVariableOp%^dense_surface/BiasAdd/ReadVariableOp$^dense_surface/MatMul/ReadVariableOp%^gru/gru_cell_2/MatMul/ReadVariableOp'^gru/gru_cell_2/MatMul_1/ReadVariableOp^gru/gru_cell_2/ReadVariableOp
^gru/while'^gru_1/gru_cell_3/MatMul/ReadVariableOp)^gru_1/gru_cell_3/MatMul_1/ReadVariableOp ^gru_1/gru_cell_3/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 2V
)dense_output/dense/BiasAdd/ReadVariableOp)dense_output/dense/BiasAdd/ReadVariableOp2T
(dense_output/dense/MatMul/ReadVariableOp(dense_output/dense/MatMul/ReadVariableOp2L
$dense_surface/BiasAdd/ReadVariableOp$dense_surface/BiasAdd/ReadVariableOp2J
#dense_surface/MatMul/ReadVariableOp#dense_surface/MatMul/ReadVariableOp2L
$gru/gru_cell_2/MatMul/ReadVariableOp$gru/gru_cell_2/MatMul/ReadVariableOp2P
&gru/gru_cell_2/MatMul_1/ReadVariableOp&gru/gru_cell_2/MatMul_1/ReadVariableOp2>
gru/gru_cell_2/ReadVariableOpgru/gru_cell_2/ReadVariableOp2
	gru/while	gru/while2P
&gru_1/gru_cell_3/MatMul/ReadVariableOp&gru_1/gru_cell_3/MatMul/ReadVariableOp2T
(gru_1/gru_cell_3/MatMul_1/ReadVariableOp(gru_1/gru_cell_3/MatMul_1/ReadVariableOp2B
gru_1/gru_cell_3/ReadVariableOpgru_1/gru_cell_3/ReadVariableOp2
gru_1/whilegru_1/while:U Q
+
_output_shapes
:€€€€€€€€€1
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1
ѓ5
К
?__inference_gru_layer_call_and_return_conditional_losses_770957

inputs#
gru_cell_2_770880:`#
gru_cell_2_770882:`#
gru_cell_2_770884: `
identity

identity_1ИҐ"gru_cell_2/StatefulPartitionedCallҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask«
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_770880gru_cell_2_770882gru_cell_2_770884*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_770879n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_770880gru_cell_2_770882gru_cell_2_770884*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_770892*
condR
while_cond_770891*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ _

Identity_1Identitywhile:output:4^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ s
NoOpNoOp#^gru_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ђ«
Ъ	
A__inference_model_layer_call_and_return_conditional_losses_772949
inputs_0
inputs_18
&gru_gru_cell_2_readvariableop_resource:`?
-gru_gru_cell_2_matmul_readvariableop_resource:`A
/gru_gru_cell_2_matmul_1_readvariableop_resource: `>
,dense_surface_matmul_readvariableop_resource:$ ;
-dense_surface_biasadd_readvariableop_resource: :
(gru_1_gru_cell_3_readvariableop_resource:`A
/gru_1_gru_cell_3_matmul_readvariableop_resource: `C
1gru_1_gru_cell_3_matmul_1_readvariableop_resource: `C
1dense_output_dense_matmul_readvariableop_resource:@@
2dense_output_dense_biasadd_readvariableop_resource:
identityИҐ)dense_output/dense/BiasAdd/ReadVariableOpҐ(dense_output/dense/MatMul/ReadVariableOpҐ$dense_surface/BiasAdd/ReadVariableOpҐ#dense_surface/MatMul/ReadVariableOpҐ$gru/gru_cell_2/MatMul/ReadVariableOpҐ&gru/gru_cell_2/MatMul_1/ReadVariableOpҐgru/gru_cell_2/ReadVariableOpҐ	gru/whileҐ&gru_1/gru_cell_3/MatMul/ReadVariableOpҐ(gru_1/gru_cell_3/MatMul_1/ReadVariableOpҐgru_1/gru_cell_3/ReadVariableOpҐgru_1/whileO
	gru/ShapeShapeinputs_0*
T0*
_output_shapes
::нѕa
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
gru/transpose	Transposeinputs_0gru/transpose/perm:output:0*
T0*+
_output_shapes
:1€€€€€€€€€Z
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
::нѕc
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ј
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“К
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“c
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:э
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskД
gru/gru_cell_2/ReadVariableOpReadVariableOp&gru_gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype0}
gru/gru_cell_2/unstackUnpack%gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numТ
$gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp-gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0Э
gru/gru_cell_2/MatMulMatMulgru/strided_slice_2:output:0,gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Х
gru/gru_cell_2/BiasAddBiasAddgru/gru_cell_2/MatMul:product:0gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`i
gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€–
gru/gru_cell_2/splitSplit'gru/gru_cell_2/split/split_dim:output:0gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЦ
&gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp/gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Ч
gru/gru_cell_2/MatMul_1MatMulgru/zeros:output:0.gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Щ
gru/gru_cell_2/BiasAdd_1BiasAdd!gru/gru_cell_2/MatMul_1:product:0gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`i
gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€k
 gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€В
gru/gru_cell_2/split_1SplitV!gru/gru_cell_2/BiasAdd_1:output:0gru/gru_cell_2/Const:output:0)gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitН
gru/gru_cell_2/addAddV2gru/gru_cell_2/split:output:0gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ k
gru/gru_cell_2/SigmoidSigmoidgru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ П
gru/gru_cell_2/add_1AddV2gru/gru_cell_2/split:output:1gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ o
gru/gru_cell_2/Sigmoid_1Sigmoidgru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ К
gru/gru_cell_2/mulMulgru/gru_cell_2/Sigmoid_1:y:0gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Ж
gru/gru_cell_2/add_2AddV2gru/gru_cell_2/split:output:2gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ g
gru/gru_cell_2/ReluRelugru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ }
gru/gru_cell_2/mul_1Mulgru/gru_cell_2/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ж
gru/gru_cell_2/subSubgru/gru_cell_2/sub/x:output:0gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ И
gru/gru_cell_2/mul_2Mulgru/gru_cell_2/sub:z:0!gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru/gru_cell_2/add_3AddV2gru/gru_cell_2/mul_1:z:0gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ƒ
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“J
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€X
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : п
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0&gru_gru_cell_2_readvariableop_resource-gru_gru_cell_2_matmul_readvariableop_resource/gru_gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *!
bodyR
gru_while_body_772681*!
condR
gru_while_cond_772680*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Е
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ќ
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:1€€€€€€€€€ *
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€e
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ы
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ґ
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€1 _
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Щ
tf.concat_3/concatConcatV2gru/while:output:4inputs_1 tf.concat_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€$Р
#dense_surface/MatMul/ReadVariableOpReadVariableOp,dense_surface_matmul_readvariableop_resource*
_output_shapes

:$ *
dtype0Ъ
dense_surface/MatMulMatMultf.concat_3/concat:output:0+dense_surface/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ О
$dense_surface/BiasAdd/ReadVariableOpReadVariableOp-dense_surface_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0†
dense_surface/BiasAddBiasAdddense_surface/MatMul:product:0,dense_surface/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ l
dense_surface/ReluReludense_surface/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€       Ь
tf.reshape_1/ReshapeReshape dense_surface/Relu:activations:0#tf.reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :≥
tf.concat_4/concatConcatV2gru/transpose_1:y:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2 d
gru_1/ShapeShapetf.concat_4/concat:output:0*
T0*
_output_shapes
::нѕc
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Е
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          О
gru_1/transpose	Transposetf.concat_4/concat:output:0gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ ^
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
::нѕe
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“^
gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ж
gru_1/ReverseV2	ReverseV2gru_1/transpose:y:0gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ М
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ч
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/ReverseV2:output:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“e
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskИ
gru_1/gru_cell_3/ReadVariableOpReadVariableOp(gru_1_gru_cell_3_readvariableop_resource*
_output_shapes

:`*
dtype0Б
gru_1/gru_cell_3/unstackUnpack'gru_1/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numЦ
&gru_1/gru_cell_3/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_3_matmul_readvariableop_resource*
_output_shapes

: `*
dtype0£
gru_1/gru_cell_3/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
gru_1/gru_cell_3/BiasAddBiasAdd!gru_1/gru_cell_3/MatMul:product:0!gru_1/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 gru_1/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
gru_1/gru_cell_3/splitSplit)gru_1/gru_cell_3/split/split_dim:output:0!gru_1/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЪ
(gru_1/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_3_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Э
gru_1/gru_cell_3/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
gru_1/gru_cell_3/BiasAdd_1BiasAdd#gru_1/gru_cell_3/MatMul_1:product:0!gru_1/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
gru_1/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"gru_1/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
gru_1/gru_cell_3/split_1SplitV#gru_1/gru_cell_3/BiasAdd_1:output:0gru_1/gru_cell_3/Const:output:0+gru_1/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
gru_1/gru_cell_3/addAddV2gru_1/gru_cell_3/split:output:0!gru_1/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
gru_1/gru_cell_3/SigmoidSigmoidgru_1/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
gru_1/gru_cell_3/add_1AddV2gru_1/gru_cell_3/split:output:1!gru_1/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
gru_1/gru_cell_3/Sigmoid_1Sigmoidgru_1/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
gru_1/gru_cell_3/mulMulgru_1/gru_cell_3/Sigmoid_1:y:0!gru_1/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
gru_1/gru_cell_3/add_2AddV2gru_1/gru_cell_3/split:output:2gru_1/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
gru_1/gru_cell_3/ReluRelugru_1/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_1/gru_cell_3/mul_1Mulgru_1/gru_cell_3/Sigmoid:y:0gru_1/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ [
gru_1/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
gru_1/gru_cell_3/subSubgru_1/gru_cell_3/sub/x:output:0gru_1/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
gru_1/gru_cell_3/mul_2Mulgru_1/gru_cell_3/sub:z:0#gru_1/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
gru_1/gru_cell_3/add_3AddV2gru_1/gru_cell_3/mul_1:z:0gru_1/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€     
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“L

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_3_readvariableop_resource/gru_1_gru_cell_3_matmul_readvariableop_resource1gru_1_gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_1_while_body_772845*#
condR
gru_1_while_cond_772844*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations З
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ‘
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€ *
element_dtype0n
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€g
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:•
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 a
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :≥
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0gru_1/transpose_1:y:0 tf.concat_5/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2@k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   У
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
(dense_output/dense/MatMul/ReadVariableOpReadVariableOp1dense_output_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¶
dense_output/dense/MatMulMatMuldense_output/Reshape:output:00dense_output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)dense_output/dense/BiasAdd/ReadVariableOpReadVariableOp2dense_output_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
dense_output/dense/BiasAddBiasAdd#dense_output/dense/MatMul:product:01dense_output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
dense_output/dense/SigmoidSigmoid#dense_output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
dense_output/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€2      Ю
dense_output/Reshape_1Reshapedense_output/dense/Sigmoid:y:0%dense_output/Reshape_1/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2m
dense_output/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Ч
dense_output/Reshape_2Reshapetf.concat_5/concat:output:0%dense_output/Reshape_2/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@r
IdentityIdentitydense_output/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2к
NoOpNoOp*^dense_output/dense/BiasAdd/ReadVariableOp)^dense_output/dense/MatMul/ReadVariableOp%^dense_surface/BiasAdd/ReadVariableOp$^dense_surface/MatMul/ReadVariableOp%^gru/gru_cell_2/MatMul/ReadVariableOp'^gru/gru_cell_2/MatMul_1/ReadVariableOp^gru/gru_cell_2/ReadVariableOp
^gru/while'^gru_1/gru_cell_3/MatMul/ReadVariableOp)^gru_1/gru_cell_3/MatMul_1/ReadVariableOp ^gru_1/gru_cell_3/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 2V
)dense_output/dense/BiasAdd/ReadVariableOp)dense_output/dense/BiasAdd/ReadVariableOp2T
(dense_output/dense/MatMul/ReadVariableOp(dense_output/dense/MatMul/ReadVariableOp2L
$dense_surface/BiasAdd/ReadVariableOp$dense_surface/BiasAdd/ReadVariableOp2J
#dense_surface/MatMul/ReadVariableOp#dense_surface/MatMul/ReadVariableOp2L
$gru/gru_cell_2/MatMul/ReadVariableOp$gru/gru_cell_2/MatMul/ReadVariableOp2P
&gru/gru_cell_2/MatMul_1/ReadVariableOp&gru/gru_cell_2/MatMul_1/ReadVariableOp2>
gru/gru_cell_2/ReadVariableOpgru/gru_cell_2/ReadVariableOp2
	gru/while	gru/while2P
&gru_1/gru_cell_3/MatMul/ReadVariableOp&gru_1/gru_cell_3/MatMul/ReadVariableOp2T
(gru_1/gru_cell_3/MatMul_1/ReadVariableOp(gru_1/gru_cell_3/MatMul_1/ReadVariableOp2B
gru_1/gru_cell_3/ReadVariableOpgru_1/gru_cell_3/ReadVariableOp2
gru_1/whilegru_1/while:U Q
+
_output_shapes
:€€€€€€€€€1
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1
тN
А
A__inference_gru_1_layer_call_and_return_conditional_losses_774169
inputs_04
"gru_cell_3_readvariableop_resource:`;
)gru_cell_3_matmul_readvariableop_resource: `=
+gru_cell_3_matmul_1_readvariableop_resource: `
identityИҐ gru_cell_3/MatMul/ReadVariableOpҐ"gru_cell_3/MatMul_1/ReadVariableOpҐgru_cell_3/ReadVariableOpҐwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    е
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask|
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource*
_output_shapes

: `*
dtype0С
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_774080*
condR
while_cond_774079*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ ≤
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs_0
Љ
™
while_cond_773705
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_773705___redundant_placeholder04
0while_while_cond_773705___redundant_placeholder14
0while_while_cond_773705___redundant_placeholder24
0while_while_cond_773705___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
≤

„
+__inference_gru_cell_3_layer_call_fn_774830

inputs
states_0
unknown:`
	unknown_0: `
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_771369o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states_0
¬ 
®
while_body_770892
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_2_770914_0:`+
while_gru_cell_2_770916_0:`+
while_gru_cell_2_770918_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_2_770914:`)
while_gru_cell_2_770916:`)
while_gru_cell_2_770918: `ИҐ(while/gru_cell_2/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0В
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_770914_0while_gru_cell_2_770916_0while_gru_cell_2_770918_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_770879Џ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: О
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w

while/NoOpNoOp)^while/gru_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_2_770914while_gru_cell_2_770914_0"4
while_gru_cell_2_770916while_gru_cell_2_770916_0"4
while_gru_cell_2_770918while_gru_cell_2_770918_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2T
(while/gru_cell_2/StatefulPartitionedCall(while/gru_cell_2/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
¬ 
®
while_body_771382
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_3_771404_0:`+
while_gru_cell_3_771406_0: `+
while_gru_cell_3_771408_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_3_771404:`)
while_gru_cell_3_771406: `)
while_gru_cell_3_771408: `ИҐ(while/gru_cell_3/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0В
(while/gru_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_3_771404_0while_gru_cell_3_771406_0while_gru_cell_3_771408_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_771369Џ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: О
while/Identity_4Identity1while/gru_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w

while/NoOpNoOp)^while/gru_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_3_771404while_gru_cell_3_771404_0"4
while_gru_cell_3_771406while_gru_cell_3_771406_0"4
while_gru_cell_3_771408while_gru_cell_3_771408_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2T
(while/gru_cell_3/StatefulPartitionedCall(while/gru_cell_3/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
€
і
&__inference_gru_1_layer_call_fn_774014

inputs
unknown:`
	unknown_0: `
	unknown_1: `
identityИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_772273s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€2 
 
_user_specified_nameinputs
Е
„
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_774763

inputs
states_0)
readvariableop_resource:`0
matmul_readvariableop_resource:`2
 matmul_1_readvariableop_resource: `
identity

identity_1ИҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:€€€€€€€€€ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ [
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states_0
†

ъ
I__inference_dense_surface_layer_call_and_return_conditional_losses_773970

inputs0
matmul_readvariableop_resource:$ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
к	
Ь
gru_1_while_cond_773177(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_773177___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_773177___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_773177___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_773177___redundant_placeholder3
gru_1_while_identity
z
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::P L

_output_shapes
: 
2
_user_specified_namegru_1/while/loop_counter:VR

_output_shapes
: 
8
_user_specified_name gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
э
’
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_771369

inputs

states)
readvariableop_resource:`0
matmul_readvariableop_resource: `2
 matmul_1_readvariableop_resource: `
identity

identity_1ИҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: `*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:€€€€€€€€€ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ [
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
Ђ
С
H__inference_dense_output_layer_call_and_return_conditional_losses_774696

inputs6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:А
	Reshape_1Reshapedense/Sigmoid:y:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€@: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Б
Ъ
-__inference_dense_output_layer_call_fn_774652

inputs
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_771551|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
„D
≠	
gru_1_while_body_773178(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0B
0gru_1_while_gru_cell_3_readvariableop_resource_0:`I
7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0: `K
9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0: `
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor@
.gru_1_while_gru_cell_3_readvariableop_resource:`G
5gru_1_while_gru_cell_3_matmul_readvariableop_resource: `I
7gru_1_while_gru_cell_3_matmul_1_readvariableop_resource: `ИҐ,gru_1/while/gru_cell_3/MatMul/ReadVariableOpҐ.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOpҐ%gru_1/while/gru_cell_3/ReadVariableOpО
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ƒ
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0Ц
%gru_1/while/gru_cell_3/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_3_readvariableop_resource_0*
_output_shapes

:`*
dtype0Н
gru_1/while/gru_cell_3/unstackUnpack-gru_1/while/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num§
,gru_1/while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: `*
dtype0«
gru_1/while/gru_cell_3/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`≠
gru_1/while/gru_cell_3/BiasAddBiasAdd'gru_1/while/gru_cell_3/MatMul:product:0'gru_1/while/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`q
&gru_1/while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€и
gru_1/while/gru_cell_3/splitSplit/gru_1/while/gru_cell_3/split/split_dim:output:0'gru_1/while/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split®
.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ѓ
gru_1/while/gru_cell_3/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`±
 gru_1/while/gru_cell_3/BiasAdd_1BiasAdd)gru_1/while/gru_cell_3/MatMul_1:product:0'gru_1/while/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`q
gru_1/while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€s
(gru_1/while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ґ
gru_1/while/gru_cell_3/split_1SplitV)gru_1/while/gru_cell_3/BiasAdd_1:output:0%gru_1/while/gru_cell_3/Const:output:01gru_1/while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split•
gru_1/while/gru_cell_3/addAddV2%gru_1/while/gru_cell_3/split:output:0'gru_1/while/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ {
gru_1/while/gru_cell_3/SigmoidSigmoidgru_1/while/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ І
gru_1/while/gru_cell_3/add_1AddV2%gru_1/while/gru_cell_3/split:output:1'gru_1/while/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 
 gru_1/while/gru_cell_3/Sigmoid_1Sigmoid gru_1/while/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ґ
gru_1/while/gru_cell_3/mulMul$gru_1/while/gru_cell_3/Sigmoid_1:y:0'gru_1/while/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Ю
gru_1/while/gru_cell_3/add_2AddV2%gru_1/while/gru_cell_3/split:output:2gru_1/while/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_1/while/gru_cell_3/ReluRelu gru_1/while/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ф
gru_1/while/gru_cell_3/mul_1Mul"gru_1/while/gru_cell_3/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ a
gru_1/while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ю
gru_1/while/gru_cell_3/subSub%gru_1/while/gru_cell_3/sub/x:output:0"gru_1/while/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ †
gru_1/while/gru_cell_3/mul_2Mulgru_1/while/gru_cell_3/sub:z:0)gru_1/while/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Ы
gru_1/while/gru_cell_3/add_3AddV2 gru_1/while/gru_cell_3/mul_1:z:0 gru_1/while/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ џ
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“S
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: В
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: Ш
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: Й
gru_1/while/Identity_4Identity gru_1/while/gru_cell_3/add_3:z:0^gru_1/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Џ
gru_1/while/NoOpNoOp-^gru_1/while/gru_cell_3/MatMul/ReadVariableOp/^gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_3_matmul_1_readvariableop_resource9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_3_matmul_readvariableop_resource7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0"b
.gru_1_while_gru_cell_3_readvariableop_resource0gru_1_while_gru_cell_3_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"ј
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2\
,gru_1/while/gru_cell_3/MatMul/ReadVariableOp,gru_1/while/gru_cell_3/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp2N
%gru_1/while/gru_cell_3/ReadVariableOp%gru_1/while/gru_cell_3/ReadVariableOp:P L

_output_shapes
: 
2
_user_specified_namegru_1/while/loop_counter:VR

_output_shapes
: 
8
_user_specified_name gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Љ
™
while_cond_771647
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_771647___redundant_placeholder04
0while_while_cond_771647___redundant_placeholder14
0while_while_cond_771647___redundant_placeholder24
0while_while_cond_771647___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
©
ґ
&__inference_gru_1_layer_call_fn_773992
inputs_0
unknown:`
	unknown_0: `
	unknown_1: `
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_771446|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs_0
вI
•

model_gru_while_body_7705440
,model_gru_while_model_gru_while_loop_counter6
2model_gru_while_model_gru_while_maximum_iterations
model_gru_while_placeholder!
model_gru_while_placeholder_1!
model_gru_while_placeholder_2/
+model_gru_while_model_gru_strided_slice_1_0k
gmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0F
4model_gru_while_gru_cell_2_readvariableop_resource_0:`M
;model_gru_while_gru_cell_2_matmul_readvariableop_resource_0:`O
=model_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0: `
model_gru_while_identity
model_gru_while_identity_1
model_gru_while_identity_2
model_gru_while_identity_3
model_gru_while_identity_4-
)model_gru_while_model_gru_strided_slice_1i
emodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensorD
2model_gru_while_gru_cell_2_readvariableop_resource:`K
9model_gru_while_gru_cell_2_matmul_readvariableop_resource:`M
;model_gru_while_gru_cell_2_matmul_1_readvariableop_resource: `ИҐ0model/gru/while/gru_cell_2/MatMul/ReadVariableOpҐ2model/gru/while/gru_cell_2/MatMul_1/ReadVariableOpҐ)model/gru/while/gru_cell_2/ReadVariableOpТ
Amodel/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ў
3model/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemgmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0model_gru_while_placeholderJmodel/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ю
)model/gru/while/gru_cell_2/ReadVariableOpReadVariableOp4model_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype0Х
"model/gru/while/gru_cell_2/unstackUnpack1model/gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numђ
0model/gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp;model_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype0”
!model/gru/while/gru_cell_2/MatMulMatMul:model/gru/while/TensorArrayV2Read/TensorListGetItem:item:08model/gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`є
"model/gru/while/gru_cell_2/BiasAddBiasAdd+model/gru/while/gru_cell_2/MatMul:product:0+model/gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`u
*model/gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ф
 model/gru/while/gru_cell_2/splitSplit3model/gru/while/gru_cell_2/split/split_dim:output:0+model/gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split∞
2model/gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp=model_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ї
#model/gru/while/gru_cell_2/MatMul_1MatMulmodel_gru_while_placeholder_2:model/gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`љ
$model/gru/while/gru_cell_2/BiasAdd_1BiasAdd-model/gru/while/gru_cell_2/MatMul_1:product:0+model/gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`u
 model/gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€w
,model/gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
"model/gru/while/gru_cell_2/split_1SplitV-model/gru/while/gru_cell_2/BiasAdd_1:output:0)model/gru/while/gru_cell_2/Const:output:05model/gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split±
model/gru/while/gru_cell_2/addAddV2)model/gru/while/gru_cell_2/split:output:0+model/gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
"model/gru/while/gru_cell_2/SigmoidSigmoid"model/gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ≥
 model/gru/while/gru_cell_2/add_1AddV2)model/gru/while/gru_cell_2/split:output:1+model/gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ З
$model/gru/while/gru_cell_2/Sigmoid_1Sigmoid$model/gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ѓ
model/gru/while/gru_cell_2/mulMul(model/gru/while/gru_cell_2/Sigmoid_1:y:0+model/gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ ™
 model/gru/while/gru_cell_2/add_2AddV2)model/gru/while/gru_cell_2/split:output:2"model/gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 
model/gru/while/gru_cell_2/ReluRelu$model/gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ †
 model/gru/while/gru_cell_2/mul_1Mul&model/gru/while/gru_cell_2/Sigmoid:y:0model_gru_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ e
 model/gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?™
model/gru/while/gru_cell_2/subSub)model/gru/while/gru_cell_2/sub/x:output:0&model/gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
 model/gru/while/gru_cell_2/mul_2Mul"model/gru/while/gru_cell_2/sub:z:0-model/gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ І
 model/gru/while/gru_cell_2/add_3AddV2$model/gru/while/gru_cell_2/mul_1:z:0$model/gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ л
4model/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmodel_gru_while_placeholder_1model_gru_while_placeholder$model/gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“W
model/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :z
model/gru/while/addAddV2model_gru_while_placeholdermodel/gru/while/add/y:output:0*
T0*
_output_shapes
: Y
model/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :П
model/gru/while/add_1AddV2,model_gru_while_model_gru_while_loop_counter model/gru/while/add_1/y:output:0*
T0*
_output_shapes
: w
model/gru/while/IdentityIdentitymodel/gru/while/add_1:z:0^model/gru/while/NoOp*
T0*
_output_shapes
: Т
model/gru/while/Identity_1Identity2model_gru_while_model_gru_while_maximum_iterations^model/gru/while/NoOp*
T0*
_output_shapes
: w
model/gru/while/Identity_2Identitymodel/gru/while/add:z:0^model/gru/while/NoOp*
T0*
_output_shapes
: §
model/gru/while/Identity_3IdentityDmodel/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/gru/while/NoOp*
T0*
_output_shapes
: Х
model/gru/while/Identity_4Identity$model/gru/while/gru_cell_2/add_3:z:0^model/gru/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ к
model/gru/while/NoOpNoOp1^model/gru/while/gru_cell_2/MatMul/ReadVariableOp3^model/gru/while/gru_cell_2/MatMul_1/ReadVariableOp*^model/gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "|
;model_gru_while_gru_cell_2_matmul_1_readvariableop_resource=model_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"x
9model_gru_while_gru_cell_2_matmul_readvariableop_resource;model_gru_while_gru_cell_2_matmul_readvariableop_resource_0"j
2model_gru_while_gru_cell_2_readvariableop_resource4model_gru_while_gru_cell_2_readvariableop_resource_0"=
model_gru_while_identity!model/gru/while/Identity:output:0"A
model_gru_while_identity_1#model/gru/while/Identity_1:output:0"A
model_gru_while_identity_2#model/gru/while/Identity_2:output:0"A
model_gru_while_identity_3#model/gru/while/Identity_3:output:0"A
model_gru_while_identity_4#model/gru/while/Identity_4:output:0"X
)model_gru_while_model_gru_strided_slice_1+model_gru_while_model_gru_strided_slice_1_0"–
emodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensorgmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2d
0model/gru/while/gru_cell_2/MatMul/ReadVariableOp0model/gru/while/gru_cell_2/MatMul/ReadVariableOp2h
2model/gru/while/gru_cell_2/MatMul_1/ReadVariableOp2model/gru/while/gru_cell_2/MatMul_1/ReadVariableOp2V
)model/gru/while/gru_cell_2/ReadVariableOp)model/gru/while/gru_cell_2/ReadVariableOp:T P

_output_shapes
: 
6
_user_specified_namemodel/gru/while/loop_counter:ZV

_output_shapes
: 
<
_user_specified_name$"model/gru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Љ
™
while_cond_773397
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_773397___redundant_placeholder04
0while_while_cond_773397___redundant_placeholder14
0while_while_cond_773397___redundant_placeholder24
0while_while_cond_773397___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Й=
щ
while_body_772009
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`C
1while_gru_cell_2_matmul_readvariableop_resource_0:`E
3while_gru_cell_2_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`A
/while_gru_cell_2_matmul_readvariableop_resource:`C
1while_gru_cell_2_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_2/MatMul/ReadVariableOpҐ(while/gru_cell_2/MatMul_1/ReadVariableOpҐwhile/gru_cell_2/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0К
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Љ
™
while_cond_771034
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_771034___redundant_placeholder04
0while_while_cond_771034___redundant_placeholder14
0while_while_cond_771034___redundant_placeholder24
0while_while_cond_771034___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Й=
щ
while_body_774235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_3_readvariableop_resource_0:`C
1while_gru_cell_3_matmul_readvariableop_resource_0: `E
3while_gru_cell_3_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_3_readvariableop_resource:`A
/while_gru_cell_3_matmul_readvariableop_resource: `C
1while_gru_cell_3_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_3/MatMul/ReadVariableOpҐ(while/gru_cell_3/MatMul_1/ReadVariableOpҐwhile/gru_cell_3/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0К
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: `*
dtype0µ
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_3/MatMul/ReadVariableOp)^while/gru_cell_3/MatMul_1/ReadVariableOp ^while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_3_matmul_1_readvariableop_resource3while_gru_cell_3_matmul_1_readvariableop_resource_0"d
/while_gru_cell_3_matmul_readvariableop_resource1while_gru_cell_3_matmul_readvariableop_resource_0"V
(while_gru_cell_3_readvariableop_resource*while_gru_cell_3_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_3/MatMul/ReadVariableOp&while/gru_cell_3/MatMul/ReadVariableOp2T
(while/gru_cell_3/MatMul_1/ReadVariableOp(while/gru_cell_3/MatMul_1/ReadVariableOp2B
while/gru_cell_3/ReadVariableOpwhile/gru_cell_3/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
к	
Ь
gru_1_while_cond_772844(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_772844___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_772844___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_772844___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_772844___redundant_placeholder3
gru_1_while_identity
z
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::P L

_output_shapes
: 
2
_user_specified_namegru_1/while/loop_counter:VR

_output_shapes
: 
8
_user_specified_name gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Й=
щ
while_body_773860
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:`C
1while_gru_cell_2_matmul_readvariableop_resource_0:`E
3while_gru_cell_2_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:`A
/while_gru_cell_2_matmul_readvariableop_resource:`C
1while_gru_cell_2_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_2/MatMul/ReadVariableOpҐ(while/gru_cell_2/MatMul_1/ReadVariableOpҐwhile/gru_cell_2/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0К
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
я	
ƒ
$__inference_gru_layer_call_fn_773295
inputs_0
unknown:`
	unknown_0:`
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_770957|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
ѓЏ
Г

!__inference__wrapped_model_770812
inputs_main

inputs_aux>
,model_gru_gru_cell_2_readvariableop_resource:`E
3model_gru_gru_cell_2_matmul_readvariableop_resource:`G
5model_gru_gru_cell_2_matmul_1_readvariableop_resource: `D
2model_dense_surface_matmul_readvariableop_resource:$ A
3model_dense_surface_biasadd_readvariableop_resource: @
.model_gru_1_gru_cell_3_readvariableop_resource:`G
5model_gru_1_gru_cell_3_matmul_readvariableop_resource: `I
7model_gru_1_gru_cell_3_matmul_1_readvariableop_resource: `I
7model_dense_output_dense_matmul_readvariableop_resource:@F
8model_dense_output_dense_biasadd_readvariableop_resource:
identityИҐ/model/dense_output/dense/BiasAdd/ReadVariableOpҐ.model/dense_output/dense/MatMul/ReadVariableOpҐ*model/dense_surface/BiasAdd/ReadVariableOpҐ)model/dense_surface/MatMul/ReadVariableOpҐ*model/gru/gru_cell_2/MatMul/ReadVariableOpҐ,model/gru/gru_cell_2/MatMul_1/ReadVariableOpҐ#model/gru/gru_cell_2/ReadVariableOpҐmodel/gru/whileҐ,model/gru_1/gru_cell_3/MatMul/ReadVariableOpҐ.model/gru_1/gru_cell_3/MatMul_1/ReadVariableOpҐ%model/gru_1/gru_cell_3/ReadVariableOpҐmodel/gru_1/whileX
model/gru/ShapeShapeinputs_main*
T0*
_output_shapes
::нѕg
model/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
model/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
model/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
model/gru/strided_sliceStridedSlicemodel/gru/Shape:output:0&model/gru/strided_slice/stack:output:0(model/gru/strided_slice/stack_1:output:0(model/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
model/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : С
model/gru/zeros/packedPack model/gru/strided_slice:output:0!model/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Z
model/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
model/gru/zerosFillmodel/gru/zeros/packed:output:0model/gru/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ m
model/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ж
model/gru/transpose	Transposeinputs_main!model/gru/transpose/perm:output:0*
T0*+
_output_shapes
:1€€€€€€€€€f
model/gru/Shape_1Shapemodel/gru/transpose:y:0*
T0*
_output_shapes
::нѕi
model/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!model/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
model/gru/strided_slice_1StridedSlicemodel/gru/Shape_1:output:0(model/gru/strided_slice_1/stack:output:0*model/gru/strided_slice_1/stack_1:output:0*model/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
%model/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€“
model/gru/TensorArrayV2TensorListReserve.model/gru/TensorArrayV2/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Р
?model/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ю
1model/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/gru/transpose:y:0Hmodel/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“i
model/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!model/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ы
model/gru/strided_slice_2StridedSlicemodel/gru/transpose:y:0(model/gru/strided_slice_2/stack:output:0*model/gru/strided_slice_2/stack_1:output:0*model/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskР
#model/gru/gru_cell_2/ReadVariableOpReadVariableOp,model_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype0Й
model/gru/gru_cell_2/unstackUnpack+model/gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numЮ
*model/gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp3model_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0ѓ
model/gru/gru_cell_2/MatMulMatMul"model/gru/strided_slice_2:output:02model/gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`І
model/gru/gru_cell_2/BiasAddBiasAdd%model/gru/gru_cell_2/MatMul:product:0%model/gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`o
$model/gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€в
model/gru/gru_cell_2/splitSplit-model/gru/gru_cell_2/split/split_dim:output:0%model/gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitҐ
,model/gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp5model_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0©
model/gru/gru_cell_2/MatMul_1MatMulmodel/gru/zeros:output:04model/gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ђ
model/gru/gru_cell_2/BiasAdd_1BiasAdd'model/gru/gru_cell_2/MatMul_1:product:0%model/gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`o
model/gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€q
&model/gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ъ
model/gru/gru_cell_2/split_1SplitV'model/gru/gru_cell_2/BiasAdd_1:output:0#model/gru/gru_cell_2/Const:output:0/model/gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЯ
model/gru/gru_cell_2/addAddV2#model/gru/gru_cell_2/split:output:0%model/gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ w
model/gru/gru_cell_2/SigmoidSigmoidmodel/gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ °
model/gru/gru_cell_2/add_1AddV2#model/gru/gru_cell_2/split:output:1%model/gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ {
model/gru/gru_cell_2/Sigmoid_1Sigmoidmodel/gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
model/gru/gru_cell_2/mulMul"model/gru/gru_cell_2/Sigmoid_1:y:0%model/gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Ш
model/gru/gru_cell_2/add_2AddV2#model/gru/gru_cell_2/split:output:2model/gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ s
model/gru/gru_cell_2/ReluRelumodel/gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ П
model/gru/gru_cell_2/mul_1Mul model/gru/gru_cell_2/Sigmoid:y:0model/gru/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ _
model/gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
model/gru/gru_cell_2/subSub#model/gru/gru_cell_2/sub/x:output:0 model/gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
model/gru/gru_cell_2/mul_2Mulmodel/gru/gru_cell_2/sub:z:0'model/gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
model/gru/gru_cell_2/add_3AddV2model/gru/gru_cell_2/mul_1:z:0model/gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ x
'model/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ÷
model/gru/TensorArrayV2_1TensorListReserve0model/gru/TensorArrayV2_1/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“P
model/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : m
"model/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€^
model/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
model/gru/whileWhile%model/gru/while/loop_counter:output:0+model/gru/while/maximum_iterations:output:0model/gru/time:output:0"model/gru/TensorArrayV2_1:handle:0model/gru/zeros:output:0"model/gru/strided_slice_1:output:0Amodel/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0,model_gru_gru_cell_2_readvariableop_resource3model_gru_gru_cell_2_matmul_readvariableop_resource5model_gru_gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *'
bodyR
model_gru_while_body_770544*'
condR
model_gru_while_cond_770543*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Л
:model/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    а
,model/gru/TensorArrayV2Stack/TensorListStackTensorListStackmodel/gru/while:output:3Cmodel/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:1€€€€€€€€€ *
element_dtype0r
model/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€k
!model/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!model/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
model/gru/strided_slice_3StridedSlice5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0(model/gru/strided_slice_3/stack:output:0*model/gru/strided_slice_3/stack_1:output:0*model/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_masko
model/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          і
model/gru/transpose_1	Transpose5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0#model/gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€1 e
model/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    _
model/tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :≠
model/tf.concat_3/concatConcatV2model/gru/while:output:4
inputs_aux&model/tf.concat_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€$Ь
)model/dense_surface/MatMul/ReadVariableOpReadVariableOp2model_dense_surface_matmul_readvariableop_resource*
_output_shapes

:$ *
dtype0ђ
model/dense_surface/MatMulMatMul!model/tf.concat_3/concat:output:01model/dense_surface/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
*model/dense_surface/BiasAdd/ReadVariableOpReadVariableOp3model_dense_surface_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0≤
model/dense_surface/BiasAddBiasAdd$model/dense_surface/MatMul:product:02model/dense_surface/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
model/dense_surface/ReluRelu$model/dense_surface/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ u
 model/tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€       Ѓ
model/tf.reshape_1/ReshapeReshape&model/dense_surface/Relu:activations:0)model/tf.reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ _
model/tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model/tf.concat_4/concatConcatV2model/gru/transpose_1:y:0#model/tf.reshape_1/Reshape:output:0&model/tf.concat_4/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2 p
model/gru_1/ShapeShape!model/tf.concat_4/concat:output:0*
T0*
_output_shapes
::нѕi
model/gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!model/gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
model/gru_1/strided_sliceStridedSlicemodel/gru_1/Shape:output:0(model/gru_1/strided_slice/stack:output:0*model/gru_1/strided_slice/stack_1:output:0*model/gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
model/gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Ч
model/gru_1/zeros/packedPack"model/gru_1/strided_slice:output:0#model/gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
model/gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Р
model/gru_1/zerosFill!model/gru_1/zeros/packed:output:0 model/gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
model/gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
model/gru_1/transpose	Transpose!model/tf.concat_4/concat:output:0#model/gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ j
model/gru_1/Shape_1Shapemodel/gru_1/transpose:y:0*
T0*
_output_shapes
::нѕk
!model/gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
model/gru_1/strided_slice_1StridedSlicemodel/gru_1/Shape_1:output:0*model/gru_1/strided_slice_1/stack:output:0,model/gru_1/strided_slice_1/stack_1:output:0,model/gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
'model/gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ў
model/gru_1/TensorArrayV2TensorListReserve0model/gru_1/TensorArrayV2/element_shape:output:0$model/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“d
model/gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ш
model/gru_1/ReverseV2	ReverseV2model/gru_1/transpose:y:0#model/gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:2€€€€€€€€€ Т
Amodel/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Й
3model/gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/gru_1/ReverseV2:output:0Jmodel/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“k
!model/gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:•
model/gru_1/strided_slice_2StridedSlicemodel/gru_1/transpose:y:0*model/gru_1/strided_slice_2/stack:output:0,model/gru_1/strided_slice_2/stack_1:output:0,model/gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskФ
%model/gru_1/gru_cell_3/ReadVariableOpReadVariableOp.model_gru_1_gru_cell_3_readvariableop_resource*
_output_shapes

:`*
dtype0Н
model/gru_1/gru_cell_3/unstackUnpack-model/gru_1/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numҐ
,model/gru_1/gru_cell_3/MatMul/ReadVariableOpReadVariableOp5model_gru_1_gru_cell_3_matmul_readvariableop_resource*
_output_shapes

: `*
dtype0µ
model/gru_1/gru_cell_3/MatMulMatMul$model/gru_1/strided_slice_2:output:04model/gru_1/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`≠
model/gru_1/gru_cell_3/BiasAddBiasAdd'model/gru_1/gru_cell_3/MatMul:product:0'model/gru_1/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`q
&model/gru_1/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€и
model/gru_1/gru_cell_3/splitSplit/model/gru_1/gru_cell_3/split/split_dim:output:0'model/gru_1/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split¶
.model/gru_1/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp7model_gru_1_gru_cell_3_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0ѓ
model/gru_1/gru_cell_3/MatMul_1MatMulmodel/gru_1/zeros:output:06model/gru_1/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`±
 model/gru_1/gru_cell_3/BiasAdd_1BiasAdd)model/gru_1/gru_cell_3/MatMul_1:product:0'model/gru_1/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`q
model/gru_1/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€s
(model/gru_1/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ґ
model/gru_1/gru_cell_3/split_1SplitV)model/gru_1/gru_cell_3/BiasAdd_1:output:0%model/gru_1/gru_cell_3/Const:output:01model/gru_1/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split•
model/gru_1/gru_cell_3/addAddV2%model/gru_1/gru_cell_3/split:output:0'model/gru_1/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ {
model/gru_1/gru_cell_3/SigmoidSigmoidmodel/gru_1/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ І
model/gru_1/gru_cell_3/add_1AddV2%model/gru_1/gru_cell_3/split:output:1'model/gru_1/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 
 model/gru_1/gru_cell_3/Sigmoid_1Sigmoid model/gru_1/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ґ
model/gru_1/gru_cell_3/mulMul$model/gru_1/gru_cell_3/Sigmoid_1:y:0'model/gru_1/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Ю
model/gru_1/gru_cell_3/add_2AddV2%model/gru_1/gru_cell_3/split:output:2model/gru_1/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ w
model/gru_1/gru_cell_3/ReluRelu model/gru_1/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
model/gru_1/gru_cell_3/mul_1Mul"model/gru_1/gru_cell_3/Sigmoid:y:0model/gru_1/zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
model/gru_1/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ю
model/gru_1/gru_cell_3/subSub%model/gru_1/gru_cell_3/sub/x:output:0"model/gru_1/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ †
model/gru_1/gru_cell_3/mul_2Mulmodel/gru_1/gru_cell_3/sub:z:0)model/gru_1/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Ы
model/gru_1/gru_cell_3/add_3AddV2 model/gru_1/gru_cell_3/mul_1:z:0 model/gru_1/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ z
)model/gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    №
model/gru_1/TensorArrayV2_1TensorListReserve2model/gru_1/TensorArrayV2_1/element_shape:output:0$model/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“R
model/gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$model/gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€`
model/gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : „
model/gru_1/whileWhile'model/gru_1/while/loop_counter:output:0-model/gru_1/while/maximum_iterations:output:0model/gru_1/time:output:0$model/gru_1/TensorArrayV2_1:handle:0model/gru_1/zeros:output:0$model/gru_1/strided_slice_1:output:0Cmodel/gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0.model_gru_1_gru_cell_3_readvariableop_resource5model_gru_1_gru_cell_3_matmul_readvariableop_resource7model_gru_1_gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *)
body!R
model_gru_1_while_body_770708*)
cond!R
model_gru_1_while_cond_770707*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Н
<model/gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ж
.model/gru_1/TensorArrayV2Stack/TensorListStackTensorListStackmodel/gru_1/while:output:3Emodel/gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€ *
element_dtype0t
!model/gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€m
#model/gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#model/gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
model/gru_1/strided_slice_3StridedSlice7model/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0*model/gru_1/strided_slice_3/stack:output:0,model/gru_1/strided_slice_3/stack_1:output:0,model/gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskq
model/gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ї
model/gru_1/transpose_1	Transpose7model/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0%model/gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 g
model/gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    _
model/tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model/tf.concat_5/concatConcatV2!model/tf.concat_4/concat:output:0model/gru_1/transpose_1:y:0&model/tf.concat_5/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2@q
 model/dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   •
model/dense_output/ReshapeReshape!model/tf.concat_5/concat:output:0)model/dense_output/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
.model/dense_output/dense/MatMul/ReadVariableOpReadVariableOp7model_dense_output_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Є
model/dense_output/dense/MatMulMatMul#model/dense_output/Reshape:output:06model/dense_output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€§
/model/dense_output/dense/BiasAdd/ReadVariableOpReadVariableOp8model_dense_output_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѕ
 model/dense_output/dense/BiasAddBiasAdd)model/dense_output/dense/MatMul:product:07model/dense_output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€И
 model/dense_output/dense/SigmoidSigmoid)model/dense_output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€w
"model/dense_output/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€2      ∞
model/dense_output/Reshape_1Reshape$model/dense_output/dense/Sigmoid:y:0+model/dense_output/Reshape_1/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2s
"model/dense_output/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ©
model/dense_output/Reshape_2Reshape!model/tf.concat_5/concat:output:0+model/dense_output/Reshape_2/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@x
IdentityIdentity%model/dense_output/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2≤
NoOpNoOp0^model/dense_output/dense/BiasAdd/ReadVariableOp/^model/dense_output/dense/MatMul/ReadVariableOp+^model/dense_surface/BiasAdd/ReadVariableOp*^model/dense_surface/MatMul/ReadVariableOp+^model/gru/gru_cell_2/MatMul/ReadVariableOp-^model/gru/gru_cell_2/MatMul_1/ReadVariableOp$^model/gru/gru_cell_2/ReadVariableOp^model/gru/while-^model/gru_1/gru_cell_3/MatMul/ReadVariableOp/^model/gru_1/gru_cell_3/MatMul_1/ReadVariableOp&^model/gru_1/gru_cell_3/ReadVariableOp^model/gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 2b
/model/dense_output/dense/BiasAdd/ReadVariableOp/model/dense_output/dense/BiasAdd/ReadVariableOp2`
.model/dense_output/dense/MatMul/ReadVariableOp.model/dense_output/dense/MatMul/ReadVariableOp2X
*model/dense_surface/BiasAdd/ReadVariableOp*model/dense_surface/BiasAdd/ReadVariableOp2V
)model/dense_surface/MatMul/ReadVariableOp)model/dense_surface/MatMul/ReadVariableOp2X
*model/gru/gru_cell_2/MatMul/ReadVariableOp*model/gru/gru_cell_2/MatMul/ReadVariableOp2\
,model/gru/gru_cell_2/MatMul_1/ReadVariableOp,model/gru/gru_cell_2/MatMul_1/ReadVariableOp2J
#model/gru/gru_cell_2/ReadVariableOp#model/gru/gru_cell_2/ReadVariableOp2"
model/gru/whilemodel/gru/while2\
,model/gru_1/gru_cell_3/MatMul/ReadVariableOp,model/gru_1/gru_cell_3/MatMul/ReadVariableOp2`
.model/gru_1/gru_cell_3/MatMul_1/ReadVariableOp.model/gru_1/gru_cell_3/MatMul_1/ReadVariableOp2N
%model/gru_1/gru_cell_3/ReadVariableOp%model/gru_1/gru_cell_3/ReadVariableOp2&
model/gru_1/whilemodel/gru_1/while:X T
+
_output_shapes
:€€€€€€€€€1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_aux
©L
б

model_gru_1_while_body_7707084
0model_gru_1_while_model_gru_1_while_loop_counter:
6model_gru_1_while_model_gru_1_while_maximum_iterations!
model_gru_1_while_placeholder#
model_gru_1_while_placeholder_1#
model_gru_1_while_placeholder_23
/model_gru_1_while_model_gru_1_strided_slice_1_0o
kmodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensor_0H
6model_gru_1_while_gru_cell_3_readvariableop_resource_0:`O
=model_gru_1_while_gru_cell_3_matmul_readvariableop_resource_0: `Q
?model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0: `
model_gru_1_while_identity 
model_gru_1_while_identity_1 
model_gru_1_while_identity_2 
model_gru_1_while_identity_3 
model_gru_1_while_identity_41
-model_gru_1_while_model_gru_1_strided_slice_1m
imodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensorF
4model_gru_1_while_gru_cell_3_readvariableop_resource:`M
;model_gru_1_while_gru_cell_3_matmul_readvariableop_resource: `O
=model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource: `ИҐ2model/gru_1/while/gru_cell_3/MatMul/ReadVariableOpҐ4model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOpҐ+model/gru_1/while/gru_cell_3/ReadVariableOpФ
Cmodel/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    в
5model/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkmodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensor_0model_gru_1_while_placeholderLmodel/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0Ґ
+model/gru_1/while/gru_cell_3/ReadVariableOpReadVariableOp6model_gru_1_while_gru_cell_3_readvariableop_resource_0*
_output_shapes

:`*
dtype0Щ
$model/gru_1/while/gru_cell_3/unstackUnpack3model/gru_1/while/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num∞
2model/gru_1/while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp=model_gru_1_while_gru_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: `*
dtype0ў
#model/gru_1/while/gru_cell_3/MatMulMatMul<model/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0:model/gru_1/while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`њ
$model/gru_1/while/gru_cell_3/BiasAddBiasAdd-model/gru_1/while/gru_cell_3/MatMul:product:0-model/gru_1/while/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`w
,model/gru_1/while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ъ
"model/gru_1/while/gru_cell_3/splitSplit5model/gru_1/while/gru_cell_3/split/split_dim:output:0-model/gru_1/while/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitі
4model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp?model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0ј
%model/gru_1/while/gru_cell_3/MatMul_1MatMulmodel_gru_1_while_placeholder_2<model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`√
&model/gru_1/while/gru_cell_3/BiasAdd_1BiasAdd/model/gru_1/while/gru_cell_3/MatMul_1:product:0-model/gru_1/while/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`w
"model/gru_1/while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€y
.model/gru_1/while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ї
$model/gru_1/while/gru_cell_3/split_1SplitV/model/gru_1/while/gru_cell_3/BiasAdd_1:output:0+model/gru_1/while/gru_cell_3/Const:output:07model/gru_1/while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЈ
 model/gru_1/while/gru_cell_3/addAddV2+model/gru_1/while/gru_cell_3/split:output:0-model/gru_1/while/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
$model/gru_1/while/gru_cell_3/SigmoidSigmoid$model/gru_1/while/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ є
"model/gru_1/while/gru_cell_3/add_1AddV2+model/gru_1/while/gru_cell_3/split:output:1-model/gru_1/while/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ Л
&model/gru_1/while/gru_cell_3/Sigmoid_1Sigmoid&model/gru_1/while/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ і
 model/gru_1/while/gru_cell_3/mulMul*model/gru_1/while/gru_cell_3/Sigmoid_1:y:0-model/gru_1/while/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ ∞
"model/gru_1/while/gru_cell_3/add_2AddV2+model/gru_1/while/gru_cell_3/split:output:2$model/gru_1/while/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
!model/gru_1/while/gru_cell_3/ReluRelu&model/gru_1/while/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ¶
"model/gru_1/while/gru_cell_3/mul_1Mul(model/gru_1/while/gru_cell_3/Sigmoid:y:0model_gru_1_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ g
"model/gru_1/while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?∞
 model/gru_1/while/gru_cell_3/subSub+model/gru_1/while/gru_cell_3/sub/x:output:0(model/gru_1/while/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ ≤
"model/gru_1/while/gru_cell_3/mul_2Mul$model/gru_1/while/gru_cell_3/sub:z:0/model/gru_1/while/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ ≠
"model/gru_1/while/gru_cell_3/add_3AddV2&model/gru_1/while/gru_cell_3/mul_1:z:0&model/gru_1/while/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ у
6model/gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmodel_gru_1_while_placeholder_1model_gru_1_while_placeholder&model/gru_1/while/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“Y
model/gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :А
model/gru_1/while/addAddV2model_gru_1_while_placeholder model/gru_1/while/add/y:output:0*
T0*
_output_shapes
: [
model/gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ч
model/gru_1/while/add_1AddV20model_gru_1_while_model_gru_1_while_loop_counter"model/gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: }
model/gru_1/while/IdentityIdentitymodel/gru_1/while/add_1:z:0^model/gru_1/while/NoOp*
T0*
_output_shapes
: Ъ
model/gru_1/while/Identity_1Identity6model_gru_1_while_model_gru_1_while_maximum_iterations^model/gru_1/while/NoOp*
T0*
_output_shapes
: }
model/gru_1/while/Identity_2Identitymodel/gru_1/while/add:z:0^model/gru_1/while/NoOp*
T0*
_output_shapes
: ™
model/gru_1/while/Identity_3IdentityFmodel/gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/gru_1/while/NoOp*
T0*
_output_shapes
: Ы
model/gru_1/while/Identity_4Identity&model/gru_1/while/gru_cell_3/add_3:z:0^model/gru_1/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ т
model/gru_1/while/NoOpNoOp3^model/gru_1/while/gru_cell_3/MatMul/ReadVariableOp5^model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp,^model/gru_1/while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "А
=model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource?model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0"|
;model_gru_1_while_gru_cell_3_matmul_readvariableop_resource=model_gru_1_while_gru_cell_3_matmul_readvariableop_resource_0"n
4model_gru_1_while_gru_cell_3_readvariableop_resource6model_gru_1_while_gru_cell_3_readvariableop_resource_0"A
model_gru_1_while_identity#model/gru_1/while/Identity:output:0"E
model_gru_1_while_identity_1%model/gru_1/while/Identity_1:output:0"E
model_gru_1_while_identity_2%model/gru_1/while/Identity_2:output:0"E
model_gru_1_while_identity_3%model/gru_1/while/Identity_3:output:0"E
model_gru_1_while_identity_4%model/gru_1/while/Identity_4:output:0"`
-model_gru_1_while_model_gru_1_strided_slice_1/model_gru_1_while_model_gru_1_strided_slice_1_0"Ў
imodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensorkmodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2h
2model/gru_1/while/gru_cell_3/MatMul/ReadVariableOp2model/gru_1/while/gru_cell_3/MatMul/ReadVariableOp2l
4model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp4model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp2Z
+model/gru_1/while/gru_cell_3/ReadVariableOp+model/gru_1/while/gru_cell_3/ReadVariableOp:V R

_output_shapes
: 
8
_user_specified_name model/gru_1/while/loop_counter:\X

_output_shapes
: 
>
_user_specified_name&$model/gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
ѓ5
К
?__inference_gru_layer_call_and_return_conditional_losses_771100

inputs#
gru_cell_2_771023:`#
gru_cell_2_771025:`#
gru_cell_2_771027: `
identity

identity_1ИҐ"gru_cell_2/StatefulPartitionedCallҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask«
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_771023gru_cell_2_771025gru_cell_2_771027*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_771022n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_771023gru_cell_2_771025gru_cell_2_771027*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_771035*
condR
while_cond_771034*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ _

Identity_1Identitywhile:output:4^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ s
NoOpNoOp#^gru_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
иM
М
?__inference_gru_layer_call_and_return_conditional_losses_773796

inputs4
"gru_cell_2_readvariableop_resource:`;
)gru_cell_2_matmul_readvariableop_resource:`=
+gru_cell_2_matmul_1_readvariableop_resource: `
identity

identity_1ИҐ gru_cell_2/MatMul/ReadVariableOpҐ"gru_cell_2/MatMul_1/ReadVariableOpҐgru_cell_2/ReadVariableOpҐwhileI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:1€€€€€€€€€R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask|
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0С
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_773706*
condR
while_cond_773705*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:1€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€1 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€1 _

Identity_1Identitywhile:output:4^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ≤
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€1: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€1
 
_user_specified_nameinputs
я

и
model_gru_while_cond_7705430
,model_gru_while_model_gru_while_loop_counter6
2model_gru_while_model_gru_while_maximum_iterations
model_gru_while_placeholder!
model_gru_while_placeholder_1!
model_gru_while_placeholder_22
.model_gru_while_less_model_gru_strided_slice_1H
Dmodel_gru_while_model_gru_while_cond_770543___redundant_placeholder0H
Dmodel_gru_while_model_gru_while_cond_770543___redundant_placeholder1H
Dmodel_gru_while_model_gru_while_cond_770543___redundant_placeholder2H
Dmodel_gru_while_model_gru_while_cond_770543___redundant_placeholder3
model_gru_while_identity
К
model/gru/while/LessLessmodel_gru_while_placeholder.model_gru_while_less_model_gru_strided_slice_1*
T0*
_output_shapes
: _
model/gru/while/IdentityIdentitymodel/gru/while/Less:z:0*
T0
*
_output_shapes
: "=
model_gru_while_identity!model/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::T P

_output_shapes
: 
6
_user_specified_namemodel/gru/while/loop_counter:ZV

_output_shapes
: 
<
_user_specified_name$"model/gru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Щ
О
model_gru_1_while_cond_7707074
0model_gru_1_while_model_gru_1_while_loop_counter:
6model_gru_1_while_model_gru_1_while_maximum_iterations!
model_gru_1_while_placeholder#
model_gru_1_while_placeholder_1#
model_gru_1_while_placeholder_26
2model_gru_1_while_less_model_gru_1_strided_slice_1L
Hmodel_gru_1_while_model_gru_1_while_cond_770707___redundant_placeholder0L
Hmodel_gru_1_while_model_gru_1_while_cond_770707___redundant_placeholder1L
Hmodel_gru_1_while_model_gru_1_while_cond_770707___redundant_placeholder2L
Hmodel_gru_1_while_model_gru_1_while_cond_770707___redundant_placeholder3
model_gru_1_while_identity
Т
model/gru_1/while/LessLessmodel_gru_1_while_placeholder2model_gru_1_while_less_model_gru_1_strided_slice_1*
T0*
_output_shapes
: c
model/gru_1/while/IdentityIdentitymodel/gru_1/while/Less:z:0*
T0
*
_output_shapes
: "A
model_gru_1_while_identity#model/gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::V R

_output_shapes
: 
8
_user_specified_name model/gru_1/while/loop_counter:\X

_output_shapes
: 
>
_user_specified_name&$model/gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Э
К
$__inference_signature_wrapper_772564

inputs_aux
inputs_main
unknown:`
	unknown_0:`
	unknown_1: `
	unknown_2:$ 
	unknown_3: 
	unknown_4:`
	unknown_5: `
	unknown_6: `
	unknown_7:@
	unknown_8:
identityИҐStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputs_main
inputs_auxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_770812s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€:€€€€€€€€€1: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_aux:XT
+
_output_shapes
:€€€€€€€€€1
%
_user_specified_nameinputs_main
µ	
¬
$__inference_gru_layer_call_fn_773334

inputs
unknown:`
	unknown_0:`
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:€€€€€€€€€1 :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_772099s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€1 q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€1
 
_user_specified_nameinputs
ТB
с
gru_while_body_772681$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0@
.gru_while_gru_cell_2_readvariableop_resource_0:`G
5gru_while_gru_cell_2_matmul_readvariableop_resource_0:`I
7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0: `
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor>
,gru_while_gru_cell_2_readvariableop_resource:`E
3gru_while_gru_cell_2_matmul_readvariableop_resource:`G
5gru_while_gru_cell_2_matmul_1_readvariableop_resource: `ИҐ*gru/while/gru_cell_2/MatMul/ReadVariableOpҐ,gru/while/gru_cell_2/MatMul_1/ReadVariableOpҐ#gru/while/gru_cell_2/ReadVariableOpМ
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ї
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Т
#gru/while/gru_cell_2/ReadVariableOpReadVariableOp.gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:`*
dtype0Й
gru/while/gru_cell_2/unstackUnpack+gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num†
*gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp5gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype0Ѕ
gru/while/gru_cell_2/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:02gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`І
gru/while/gru_cell_2/BiasAddBiasAdd%gru/while/gru_cell_2/MatMul:product:0%gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`o
$gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€в
gru/while/gru_cell_2/splitSplit-gru/while/gru_cell_2/split/split_dim:output:0%gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split§
,gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0®
gru/while/gru_cell_2/MatMul_1MatMulgru_while_placeholder_24gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ђ
gru/while/gru_cell_2/BiasAdd_1BiasAdd'gru/while/gru_cell_2/MatMul_1:product:0%gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`o
gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€q
&gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ъ
gru/while/gru_cell_2/split_1SplitV'gru/while/gru_cell_2/BiasAdd_1:output:0#gru/while/gru_cell_2/Const:output:0/gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЯ
gru/while/gru_cell_2/addAddV2#gru/while/gru_cell_2/split:output:0%gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru/while/gru_cell_2/SigmoidSigmoidgru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ °
gru/while/gru_cell_2/add_1AddV2#gru/while/gru_cell_2/split:output:1%gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ {
gru/while/gru_cell_2/Sigmoid_1Sigmoidgru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
gru/while/gru_cell_2/mulMul"gru/while/gru_cell_2/Sigmoid_1:y:0%gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Ш
gru/while/gru_cell_2/add_2AddV2#gru/while/gru_cell_2/split:output:2gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ s
gru/while/gru_cell_2/ReluRelugru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ О
gru/while/gru_cell_2/mul_1Mul gru/while/gru_cell_2/Sigmoid:y:0gru_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ _
gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
gru/while/gru_cell_2/subSub#gru/while/gru_cell_2/sub/x:output:0 gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
gru/while/gru_cell_2/mul_2Mulgru/while/gru_cell_2/sub:z:0'gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
gru/while/gru_cell_2/add_3AddV2gru/while/gru_cell_2/mul_1:z:0gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“Q
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: Т
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: Г
gru/while/Identity_4Identitygru/while/gru_cell_2/add_3:z:0^gru/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ “
gru/while/NoOpNoOp+^gru/while/gru_cell_2/MatMul/ReadVariableOp-^gru/while/gru_cell_2/MatMul_1/ReadVariableOp$^gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "p
5gru_while_gru_cell_2_matmul_1_readvariableop_resource7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"l
3gru_while_gru_cell_2_matmul_readvariableop_resource5gru_while_gru_cell_2_matmul_readvariableop_resource_0"^
,gru_while_gru_cell_2_readvariableop_resource.gru_while_gru_cell_2_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"Є
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2X
*gru/while/gru_cell_2/MatMul/ReadVariableOp*gru/while/gru_cell_2/MatMul/ReadVariableOp2\
,gru/while/gru_cell_2/MatMul_1/ReadVariableOp,gru/while/gru_cell_2/MatMul_1/ReadVariableOp2J
#gru/while/gru_cell_2/ReadVariableOp#gru/while/gru_cell_2/ReadVariableOp:N J

_output_shapes
: 
0
_user_specified_namegru/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namegru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
„D
≠	
gru_1_while_body_772845(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0B
0gru_1_while_gru_cell_3_readvariableop_resource_0:`I
7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0: `K
9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0: `
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor@
.gru_1_while_gru_cell_3_readvariableop_resource:`G
5gru_1_while_gru_cell_3_matmul_readvariableop_resource: `I
7gru_1_while_gru_cell_3_matmul_1_readvariableop_resource: `ИҐ,gru_1/while/gru_cell_3/MatMul/ReadVariableOpҐ.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOpҐ%gru_1/while/gru_cell_3/ReadVariableOpО
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ƒ
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0Ц
%gru_1/while/gru_cell_3/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_3_readvariableop_resource_0*
_output_shapes

:`*
dtype0Н
gru_1/while/gru_cell_3/unstackUnpack-gru_1/while/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num§
,gru_1/while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: `*
dtype0«
gru_1/while/gru_cell_3/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`≠
gru_1/while/gru_cell_3/BiasAddBiasAdd'gru_1/while/gru_cell_3/MatMul:product:0'gru_1/while/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`q
&gru_1/while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€и
gru_1/while/gru_cell_3/splitSplit/gru_1/while/gru_cell_3/split/split_dim:output:0'gru_1/while/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split®
.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ѓ
gru_1/while/gru_cell_3/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`±
 gru_1/while/gru_cell_3/BiasAdd_1BiasAdd)gru_1/while/gru_cell_3/MatMul_1:product:0'gru_1/while/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`q
gru_1/while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€s
(gru_1/while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ґ
gru_1/while/gru_cell_3/split_1SplitV)gru_1/while/gru_cell_3/BiasAdd_1:output:0%gru_1/while/gru_cell_3/Const:output:01gru_1/while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split•
gru_1/while/gru_cell_3/addAddV2%gru_1/while/gru_cell_3/split:output:0'gru_1/while/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ {
gru_1/while/gru_cell_3/SigmoidSigmoidgru_1/while/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ І
gru_1/while/gru_cell_3/add_1AddV2%gru_1/while/gru_cell_3/split:output:1'gru_1/while/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 
 gru_1/while/gru_cell_3/Sigmoid_1Sigmoid gru_1/while/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ґ
gru_1/while/gru_cell_3/mulMul$gru_1/while/gru_cell_3/Sigmoid_1:y:0'gru_1/while/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Ю
gru_1/while/gru_cell_3/add_2AddV2%gru_1/while/gru_cell_3/split:output:2gru_1/while/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_1/while/gru_cell_3/ReluRelu gru_1/while/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Ф
gru_1/while/gru_cell_3/mul_1Mul"gru_1/while/gru_cell_3/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ a
gru_1/while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ю
gru_1/while/gru_cell_3/subSub%gru_1/while/gru_cell_3/sub/x:output:0"gru_1/while/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ †
gru_1/while/gru_cell_3/mul_2Mulgru_1/while/gru_cell_3/sub:z:0)gru_1/while/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Ы
gru_1/while/gru_cell_3/add_3AddV2 gru_1/while/gru_cell_3/mul_1:z:0 gru_1/while/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ џ
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“S
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: В
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: Ш
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: Й
gru_1/while/Identity_4Identity gru_1/while/gru_cell_3/add_3:z:0^gru_1/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Џ
gru_1/while/NoOpNoOp-^gru_1/while/gru_cell_3/MatMul/ReadVariableOp/^gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_3_matmul_1_readvariableop_resource9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_3_matmul_readvariableop_resource7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0"b
.gru_1_while_gru_cell_3_readvariableop_resource0gru_1_while_gru_cell_3_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"ј
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2\
,gru_1/while/gru_cell_3/MatMul/ReadVariableOp,gru_1/while/gru_cell_3/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp2N
%gru_1/while/gru_cell_3/ReadVariableOp%gru_1/while/gru_cell_3/ReadVariableOp:P L

_output_shapes
: 
2
_user_specified_namegru_1/while/loop_counter:VR

_output_shapes
: 
8
_user_specified_name gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Ђ
С
H__inference_dense_output_layer_call_and_return_conditional_losses_774674

inputs6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:А
	Reshape_1Reshapedense/Sigmoid:y:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€@: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Љ
™
while_cond_774234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_774234___redundant_placeholder04
0while_while_cond_774234___redundant_placeholder14
0while_while_cond_774234___redundant_placeholder24
0while_while_cond_774234___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Е
„
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_774802

inputs
states_0)
readvariableop_resource:`0
matmul_readvariableop_resource:`2
 matmul_1_readvariableop_resource: `
identity

identity_1ИҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:€€€€€€€€€ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ [
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states_0
я	
ƒ
$__inference_gru_layer_call_fn_773308
inputs_0
unknown:`
	unknown_0:`
	unknown_1: `
identity

identity_1ИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_771100|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
б#
“
A__inference_model_layer_call_and_return_conditional_losses_772291
inputs_main

inputs_aux

gru_772100:`

gru_772102:`

gru_772104: `&
dense_surface_772110:$ "
dense_surface_772112: 
gru_1_772274:`
gru_1_772276: `
gru_1_772278: `%
dense_output_772283:@!
dense_output_772285:
identityИҐ$dense_output/StatefulPartitionedCallҐ%dense_surface/StatefulPartitionedCallҐgru/StatefulPartitionedCallҐgru_1/StatefulPartitionedCallК
gru/StatefulPartitionedCallStatefulPartitionedCallinputs_main
gru_772100
gru_772102
gru_772104*
Tin
2*
Tout
2*
_collective_manager_ids
 *>
_output_shapes,
*:€€€€€€€€€1 :€€€€€€€€€ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_772099Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :≠
tf.concat_3/concatConcatV2$gru/StatefulPartitionedCall:output:1
inputs_aux tf.concat_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€$Ь
%dense_surface/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_surface_772110dense_surface_772112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_771760o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€       ™
tf.reshape_1/ReshapeReshape.dense_surface/StatefulPartitionedCall:output:0#tf.reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ƒ
tf.concat_4/concatConcatV2$gru/StatefulPartitionedCall:output:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2 Р
gru_1/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0gru_1_772274gru_1_772276gru_1_772278*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_772273Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ƒ
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0&gru_1/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€2@Ь
$dense_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_output_772283dense_output_772285*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_771551k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   У
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@А
IdentityIdentity-dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2”
NoOpNoOp%^dense_output/StatefulPartitionedCall&^dense_surface/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€1:€€€€€€€€€: : : : : : : : : : 2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall2N
%dense_surface/StatefulPartitionedCall%dense_surface/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:X T
+
_output_shapes
:€€€€€€€€€1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_aux
Ч

т
A__inference_dense_layer_call_and_return_conditional_losses_774928

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Й=
щ
while_body_772184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_3_readvariableop_resource_0:`C
1while_gru_cell_3_matmul_readvariableop_resource_0: `E
3while_gru_cell_3_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_3_readvariableop_resource:`A
/while_gru_cell_3_matmul_readvariableop_resource: `C
1while_gru_cell_3_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_3/MatMul/ReadVariableOpҐ(while/gru_cell_3/MatMul_1/ReadVariableOpҐwhile/gru_cell_3/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0К
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: `*
dtype0µ
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_3/MatMul/ReadVariableOp)^while/gru_cell_3/MatMul_1/ReadVariableOp ^while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_3_matmul_1_readvariableop_resource3while_gru_cell_3_matmul_1_readvariableop_resource_0"d
/while_gru_cell_3_matmul_readvariableop_resource1while_gru_cell_3_matmul_readvariableop_resource_0"V
(while_gru_cell_3_readvariableop_resource*while_gru_cell_3_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_3/MatMul/ReadVariableOp&while/gru_cell_3/MatMul/ReadVariableOp2T
(while/gru_cell_3/MatMul_1/ReadVariableOp(while/gru_cell_3/MatMul_1/ReadVariableOp2B
while/gru_cell_3/ReadVariableOpwhile/gru_cell_3/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
¶N
О
?__inference_gru_layer_call_and_return_conditional_losses_773642
inputs_04
"gru_cell_2_readvariableop_resource:`;
)gru_cell_2_matmul_readvariableop_resource:`=
+gru_cell_2_matmul_1_readvariableop_resource: `
identity

identity_1ИҐ gru_cell_2/MatMul/ReadVariableOpҐ"gru_cell_2/MatMul_1/ReadVariableOpҐgru_cell_2/ReadVariableOpҐwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::нѕ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask|
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:`*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numК
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0С
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Й
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ƒ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitО
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0Л
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Н
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€т
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitБ
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:€€€€€€€€€ U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ |
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_773552*
condR
while_cond_773551*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ _

Identity_1Identitywhile:output:4^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ≤
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
Й=
щ
while_body_774080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_3_readvariableop_resource_0:`C
1while_gru_cell_3_matmul_readvariableop_resource_0: `E
3while_gru_cell_3_matmul_1_readvariableop_resource_0: `
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_3_readvariableop_resource:`A
/while_gru_cell_3_matmul_readvariableop_resource: `C
1while_gru_cell_3_matmul_1_readvariableop_resource: `ИҐ&while/gru_cell_3/MatMul/ReadVariableOpҐ(while/gru_cell_3/MatMul_1/ReadVariableOpҐwhile/gru_cell_3/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0К
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes

:`*
dtype0Б
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numШ
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0*
_output_shapes

: `*
dtype0µ
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Ы
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€÷
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitЬ
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype0Ь
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`Я
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€К
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitУ
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ s
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Р
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ М
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ В
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:€€€€€€€€€ [
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?М
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ О
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ √
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ¬

while/NoOpNoOp'^while/gru_cell_3/MatMul/ReadVariableOp)^while/gru_cell_3/MatMul_1/ReadVariableOp ^while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_3_matmul_1_readvariableop_resource3while_gru_cell_3_matmul_1_readvariableop_resource_0"d
/while_gru_cell_3_matmul_readvariableop_resource1while_gru_cell_3_matmul_readvariableop_resource_0"V
(while_gru_cell_3_readvariableop_resource*while_gru_cell_3_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2P
&while/gru_cell_3/MatMul/ReadVariableOp&while/gru_cell_3/MatMul/ReadVariableOp2T
(while/gru_cell_3/MatMul_1/ReadVariableOp(while/gru_cell_3/MatMul_1/ReadVariableOp2B
while/gru_cell_3/ReadVariableOpwhile/gru_cell_3/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Е
„
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_774908

inputs
states_0)
readvariableop_resource:`0
matmul_readvariableop_resource: `2
 matmul_1_readvariableop_resource: `
identity

identity_1ИҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpҐReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: `*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:€€€€€€€€€`Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:€€€€€€€€€`Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"        €€€€\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€∆
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€ I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:€€€€€€€€€ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€ [
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€ : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states_0"у
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*В
serving_defaultо
A

inputs_aux3
serving_default_inputs_aux:0€€€€€€€€€
G
inputs_main8
serving_default_inputs_main:0€€€€€€€€€1D
dense_output4
StatefulPartitionedCall:0€€€€€€€€€2tensorflow/serving/predict:°Ч
„
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Џ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
ї
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
(
'	keras_api"
_tf_keras_layer
(
(	keras_api"
_tf_keras_layer
Џ
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/_random_generator
0cell
1
state_spec"
_tf_keras_rnn_layer
(
2	keras_api"
_tf_keras_layer
∞
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
	9layer"
_tf_keras_layer
f
:0
;1
<2
%3
&4
=5
>6
?7
@8
A9"
trackable_list_wrapper
f
:0
;1
<2
%3
&4
=5
>6
?7
@8
A9"
trackable_list_wrapper
 "
trackable_list_wrapper
 
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
√
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32Ў
&__inference_model_layer_call_fn_772358
&__inference_model_layer_call_fn_772424
&__inference_model_layer_call_fn_772590
&__inference_model_layer_call_fn_772616µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
ѓ
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32ƒ
A__inference_model_layer_call_and_return_conditional_losses_771942
A__inference_model_layer_call_and_return_conditional_losses_772291
A__inference_model_layer_call_and_return_conditional_losses_772949
A__inference_model_layer_call_and_return_conditional_losses_773282µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
№Bў
!__inference__wrapped_model_770812inputs_main
inputs_aux"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
j
O
_variables
P_iterations
Q_learning_rate
R_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
Sserving_default"
signature_map
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
є

Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
–
Ztrace_0
[trace_1
\trace_2
]trace_32е
$__inference_gru_layer_call_fn_773295
$__inference_gru_layer_call_fn_773308
$__inference_gru_layer_call_fn_773321
$__inference_gru_layer_call_fn_773334 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zZtrace_0z[trace_1z\trace_2z]trace_3
Љ
^trace_0
_trace_1
`trace_2
atrace_32—
?__inference_gru_layer_call_and_return_conditional_losses_773488
?__inference_gru_layer_call_and_return_conditional_losses_773642
?__inference_gru_layer_call_and_return_conditional_losses_773796
?__inference_gru_layer_call_and_return_conditional_losses_773950 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z^trace_0z_trace_1z`trace_2zatrace_3
"
_generic_user_object
и
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator

:kernel
;recurrent_kernel
<bias"
_tf_keras_layer
 "
trackable_list_wrapper
"
_generic_user_object
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
и
ntrace_02Ћ
.__inference_dense_surface_layer_call_fn_773959Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zntrace_0
Г
otrace_02ж
I__inference_dense_surface_layer_call_and_return_conditional_losses_773970Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zotrace_0
&:$$ 2dense_surface/kernel
 : 2dense_surface/bias
"
_generic_user_object
"
_generic_user_object
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
є

pstates
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Ў
vtrace_0
wtrace_1
xtrace_2
ytrace_32н
&__inference_gru_1_layer_call_fn_773981
&__inference_gru_1_layer_call_fn_773992
&__inference_gru_1_layer_call_fn_774003
&__inference_gru_1_layer_call_fn_774014 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zvtrace_0zwtrace_1zxtrace_2zytrace_3
ƒ
ztrace_0
{trace_1
|trace_2
}trace_32ў
A__inference_gru_1_layer_call_and_return_conditional_losses_774169
A__inference_gru_1_layer_call_and_return_conditional_losses_774324
A__inference_gru_1_layer_call_and_return_conditional_losses_774479
A__inference_gru_1_layer_call_and_return_conditional_losses_774634 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zztrace_0z{trace_1z|trace_2z}trace_3
"
_generic_user_object
н
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses
Д_random_generator

=kernel
>recurrent_kernel
?bias"
_tf_keras_layer
 "
trackable_list_wrapper
"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
—
Кtrace_0
Лtrace_12Ц
-__inference_dense_output_layer_call_fn_774643
-__inference_dense_output_layer_call_fn_774652µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zКtrace_0zЛtrace_1
З
Мtrace_0
Нtrace_12ћ
H__inference_dense_output_layer_call_and_return_conditional_losses_774674
H__inference_dense_output_layer_call_and_return_conditional_losses_774696µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zМtrace_0zНtrace_1
Ѕ
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
':%`2gru/gru_cell_2/kernel
1:/ `2gru/gru_cell_2/recurrent_kernel
%:#`2gru/gru_cell_2/bias
):' `2gru_1/gru_cell_3/kernel
3:1 `2!gru_1/gru_cell_3/recurrent_kernel
':%`2gru_1/gru_cell_3/bias
%:#@2dense_output/kernel
:2dense_output/bias
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
8
Ф0
Х1
Ц2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
&__inference_model_layer_call_fn_772358inputs_main
inputs_aux"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
&__inference_model_layer_call_fn_772424inputs_main
inputs_aux"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
&__inference_model_layer_call_fn_772590inputs_0inputs_1"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
&__inference_model_layer_call_fn_772616inputs_0inputs_1"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
A__inference_model_layer_call_and_return_conditional_losses_771942inputs_main
inputs_aux"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
A__inference_model_layer_call_and_return_conditional_losses_772291inputs_main
inputs_aux"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ФBС
A__inference_model_layer_call_and_return_conditional_losses_772949inputs_0inputs_1"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ФBС
A__inference_model_layer_call_and_return_conditional_losses_773282inputs_0inputs_1"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
'
P0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
µ2≤ѓ
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
ўB÷
$__inference_signature_wrapper_772564
inputs_auxinputs_main"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ВB€
$__inference_gru_layer_call_fn_773295inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
$__inference_gru_layer_call_fn_773308inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
$__inference_gru_layer_call_fn_773321inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
$__inference_gru_layer_call_fn_773334inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЭBЪ
?__inference_gru_layer_call_and_return_conditional_losses_773488inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЭBЪ
?__inference_gru_layer_call_and_return_conditional_losses_773642inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
?__inference_gru_layer_call_and_return_conditional_losses_773796inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
?__inference_gru_layer_call_and_return_conditional_losses_773950inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
Ћ
Ьtrace_0
Эtrace_12Р
+__inference_gru_cell_2_layer_call_fn_774710
+__inference_gru_cell_2_layer_call_fn_774724≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЬtrace_0zЭtrace_1
Б
Юtrace_0
Яtrace_12∆
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_774763
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_774802≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЮtrace_0zЯtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЎB’
.__inference_dense_surface_layer_call_fn_773959inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_dense_surface_layer_call_and_return_conditional_losses_773970inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ДBБ
&__inference_gru_1_layer_call_fn_773981inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ДBБ
&__inference_gru_1_layer_call_fn_773992inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
&__inference_gru_1_layer_call_fn_774003inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
&__inference_gru_1_layer_call_fn_774014inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЯBЬ
A__inference_gru_1_layer_call_and_return_conditional_losses_774169inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЯBЬ
A__inference_gru_1_layer_call_and_return_conditional_losses_774324inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЭBЪ
A__inference_gru_1_layer_call_and_return_conditional_losses_774479inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЭBЪ
A__inference_gru_1_layer_call_and_return_conditional_losses_774634inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
ґ
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
~	variables
trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
Ћ
•trace_0
¶trace_12Р
+__inference_gru_cell_3_layer_call_fn_774816
+__inference_gru_cell_3_layer_call_fn_774830≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z•trace_0z¶trace_1
Б
Іtrace_0
®trace_12∆
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_774869
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_774908≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zІtrace_0z®trace_1
"
_generic_user_object
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
-__inference_dense_output_layer_call_fn_774643inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
фBс
-__inference_dense_output_layer_call_fn_774652inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
H__inference_dense_output_layer_call_and_return_conditional_losses_774674inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
H__inference_dense_output_layer_call_and_return_conditional_losses_774696inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
©non_trainable_variables
™layers
Ђmetrics
 ђlayer_regularization_losses
≠layer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
в
Ѓtrace_02√
&__inference_dense_layer_call_fn_774917Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЃtrace_0
э
ѓtrace_02ё
A__inference_dense_layer_call_and_return_conditional_losses_774928Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zѓtrace_0
R
∞	variables
±	keras_api

≤total

≥count"
_tf_keras_metric
c
і	variables
µ	keras_api

ґtotal

Јcount
Є
_fn_kwargs"
_tf_keras_metric
c
є	variables
Ї	keras_api

їtotal

Љcount
љ
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
+__inference_gru_cell_2_layer_call_fn_774710inputsstates_0"≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
+__inference_gru_cell_2_layer_call_fn_774724inputsstates_0"≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ХBТ
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_774763inputsstates_0"≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ХBТ
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_774802inputsstates_0"≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
+__inference_gru_cell_3_layer_call_fn_774816inputsstates_0"≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
+__inference_gru_cell_3_layer_call_fn_774830inputsstates_0"≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ХBТ
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_774869inputsstates_0"≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ХBТ
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_774908inputsstates_0"≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
–BЌ
&__inference_dense_layer_call_fn_774917inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
лBи
A__inference_dense_layer_call_and_return_conditional_losses_774928inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
≤0
≥1"
trackable_list_wrapper
.
∞	variables"
_generic_user_object
:  (2total
:  (2count
0
ґ0
Ј1"
trackable_list_wrapper
.
і	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ї0
Љ1"
trackable_list_wrapper
.
є	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЎ
!__inference__wrapped_model_770812≤
<:;%&?=>@AcҐ`
YҐV
TЪQ
)К&
inputs_main€€€€€€€€€1
$К!

inputs_aux€€€€€€€€€
™ "?™<
:
dense_output*К'
dense_output€€€€€€€€€2®
A__inference_dense_layer_call_and_return_conditional_losses_774928c@A/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ В
&__inference_dense_layer_call_fn_774917X@A/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "!К
unknown€€€€€€€€€“
H__inference_dense_output_layer_call_and_return_conditional_losses_774674Е@ADҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€@
p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ “
H__inference_dense_output_layer_call_and_return_conditional_losses_774696Е@ADҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ Ђ
-__inference_dense_output_layer_call_fn_774643z@ADҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€@
p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€Ђ
-__inference_dense_output_layer_call_fn_774652z@ADҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€∞
I__inference_dense_surface_layer_call_and_return_conditional_losses_773970c%&/Ґ,
%Ґ"
 К
inputs€€€€€€€€€$
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ К
.__inference_dense_surface_layer_call_fn_773959X%&/Ґ,
%Ґ"
 К
inputs€€€€€€€€€$
™ "!К
unknown€€€€€€€€€ „
A__inference_gru_1_layer_call_and_return_conditional_losses_774169С?=>OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€ 

 
p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€ 
Ъ „
A__inference_gru_1_layer_call_and_return_conditional_losses_774324С?=>OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€ 

 
p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€ 
Ъ љ
A__inference_gru_1_layer_call_and_return_conditional_losses_774479x?=>?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€2 

 
p

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2 
Ъ љ
A__inference_gru_1_layer_call_and_return_conditional_losses_774634x?=>?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€2 

 
p 

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2 
Ъ ±
&__inference_gru_1_layer_call_fn_773981Ж?=>OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€ 

 
p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€ ±
&__inference_gru_1_layer_call_fn_773992Ж?=>OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€ 

 
p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€ Ч
&__inference_gru_1_layer_call_fn_774003m?=>?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€2 

 
p

 
™ "%К"
unknown€€€€€€€€€2 Ч
&__inference_gru_1_layer_call_fn_774014m?=>?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€2 

 
p 

 
™ "%К"
unknown€€€€€€€€€2 Р
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_774763≈<:;\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states_0€€€€€€€€€ 
p
™ "`Ґ]
VҐS
$К!

tensor_0_0€€€€€€€€€ 
+Ъ(
&К#
tensor_0_1_0€€€€€€€€€ 
Ъ Р
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_774802≈<:;\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states_0€€€€€€€€€ 
p 
™ "`Ґ]
VҐS
$К!

tensor_0_0€€€€€€€€€ 
+Ъ(
&К#
tensor_0_1_0€€€€€€€€€ 
Ъ з
+__inference_gru_cell_2_layer_call_fn_774710Ј<:;\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states_0€€€€€€€€€ 
p
™ "RҐO
"К
tensor_0€€€€€€€€€ 
)Ъ&
$К!

tensor_1_0€€€€€€€€€ з
+__inference_gru_cell_2_layer_call_fn_774724Ј<:;\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states_0€€€€€€€€€ 
p 
™ "RҐO
"К
tensor_0€€€€€€€€€ 
)Ъ&
$К!

tensor_1_0€€€€€€€€€ Р
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_774869≈?=>\ҐY
RҐO
 К
inputs€€€€€€€€€ 
'Ґ$
"К
states_0€€€€€€€€€ 
p
™ "`Ґ]
VҐS
$К!

tensor_0_0€€€€€€€€€ 
+Ъ(
&К#
tensor_0_1_0€€€€€€€€€ 
Ъ Р
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_774908≈?=>\ҐY
RҐO
 К
inputs€€€€€€€€€ 
'Ґ$
"К
states_0€€€€€€€€€ 
p 
™ "`Ґ]
VҐS
$К!

tensor_0_0€€€€€€€€€ 
+Ъ(
&К#
tensor_0_1_0€€€€€€€€€ 
Ъ з
+__inference_gru_cell_3_layer_call_fn_774816Ј?=>\ҐY
RҐO
 К
inputs€€€€€€€€€ 
'Ґ$
"К
states_0€€€€€€€€€ 
p
™ "RҐO
"К
tensor_0€€€€€€€€€ 
)Ъ&
$К!

tensor_1_0€€€€€€€€€ з
+__inference_gru_cell_3_layer_call_fn_774830Ј?=>\ҐY
RҐO
 К
inputs€€€€€€€€€ 
'Ґ$
"К
states_0€€€€€€€€€ 
p 
™ "RҐO
"К
tensor_0€€€€€€€€€ 
)Ъ&
$К!

tensor_1_0€€€€€€€€€ В
?__inference_gru_layer_call_and_return_conditional_losses_773488Њ<:;OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p

 
™ "fҐc
\ЪY
1К.

tensor_0_0€€€€€€€€€€€€€€€€€€ 
$К!

tensor_0_1€€€€€€€€€ 
Ъ В
?__inference_gru_layer_call_and_return_conditional_losses_773642Њ<:;OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "fҐc
\ЪY
1К.

tensor_0_0€€€€€€€€€€€€€€€€€€ 
$К!

tensor_0_1€€€€€€€€€ 
Ъ й
?__inference_gru_layer_call_and_return_conditional_losses_773796•<:;?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€1

 
p

 
™ "]ҐZ
SЪP
(К%

tensor_0_0€€€€€€€€€1 
$К!

tensor_0_1€€€€€€€€€ 
Ъ й
?__inference_gru_layer_call_and_return_conditional_losses_773950•<:;?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€1

 
p 

 
™ "]ҐZ
SЪP
(К%

tensor_0_0€€€€€€€€€1 
$К!

tensor_0_1€€€€€€€€€ 
Ъ ў
$__inference_gru_layer_call_fn_773295∞<:;OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p

 
™ "XЪU
/К,
tensor_0€€€€€€€€€€€€€€€€€€ 
"К
tensor_1€€€€€€€€€ ў
$__inference_gru_layer_call_fn_773308∞<:;OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "XЪU
/К,
tensor_0€€€€€€€€€€€€€€€€€€ 
"К
tensor_1€€€€€€€€€ ј
$__inference_gru_layer_call_fn_773321Ч<:;?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€1

 
p

 
™ "OЪL
&К#
tensor_0€€€€€€€€€1 
"К
tensor_1€€€€€€€€€ ј
$__inference_gru_layer_call_fn_773334Ч<:;?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€1

 
p 

 
™ "OЪL
&К#
tensor_0€€€€€€€€€1 
"К
tensor_1€€€€€€€€€ с
A__inference_model_layer_call_and_return_conditional_losses_771942Ђ
<:;%&?=>@AkҐh
aҐ^
TЪQ
)К&
inputs_main€€€€€€€€€1
$К!

inputs_aux€€€€€€€€€
p

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2
Ъ с
A__inference_model_layer_call_and_return_conditional_losses_772291Ђ
<:;%&?=>@AkҐh
aҐ^
TЪQ
)К&
inputs_main€€€€€€€€€1
$К!

inputs_aux€€€€€€€€€
p 

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2
Ъ м
A__inference_model_layer_call_and_return_conditional_losses_772949¶
<:;%&?=>@AfҐc
\ҐY
OЪL
&К#
inputs_0€€€€€€€€€1
"К
inputs_1€€€€€€€€€
p

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2
Ъ м
A__inference_model_layer_call_and_return_conditional_losses_773282¶
<:;%&?=>@AfҐc
\ҐY
OЪL
&К#
inputs_0€€€€€€€€€1
"К
inputs_1€€€€€€€€€
p 

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2
Ъ Ћ
&__inference_model_layer_call_fn_772358†
<:;%&?=>@AkҐh
aҐ^
TЪQ
)К&
inputs_main€€€€€€€€€1
$К!

inputs_aux€€€€€€€€€
p

 
™ "%К"
unknown€€€€€€€€€2Ћ
&__inference_model_layer_call_fn_772424†
<:;%&?=>@AkҐh
aҐ^
TЪQ
)К&
inputs_main€€€€€€€€€1
$К!

inputs_aux€€€€€€€€€
p 

 
™ "%К"
unknown€€€€€€€€€2∆
&__inference_model_layer_call_fn_772590Ы
<:;%&?=>@AfҐc
\ҐY
OЪL
&К#
inputs_0€€€€€€€€€1
"К
inputs_1€€€€€€€€€
p

 
™ "%К"
unknown€€€€€€€€€2∆
&__inference_model_layer_call_fn_772616Ы
<:;%&?=>@AfҐc
\ҐY
OЪL
&К#
inputs_0€€€€€€€€€1
"К
inputs_1€€€€€€€€€
p 

 
™ "%К"
unknown€€€€€€€€€2у
$__inference_signature_wrapper_772564 
<:;%&?=>@A{Ґx
Ґ 
q™n
2

inputs_aux$К!

inputs_aux€€€€€€€€€
8
inputs_main)К&
inputs_main€€€€€€€€€1"?™<
:
dense_output*К'
dense_output€€€€€€€€€2