¶ž#
ę
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

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
resource
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
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
list(type)(0
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
output"out_typeķout_type"	
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

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
Į
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
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
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
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint’’’’’’’’’
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758ģÄ!
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

dense_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_namedense_output/kernel
|
'dense_output/kernel/Read/ReadVariableOpReadVariableOpdense_output/kernel*
_output_shapes
:	*
dtype0

gru_1/gru_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_namegru_1/gru_cell_3/bias

)gru_1/gru_cell_3/bias/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_3/bias*
_output_shapes
:	*
dtype0
 
!gru_1/gru_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!gru_1/gru_cell_3/recurrent_kernel

5gru_1/gru_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_1/gru_cell_3/recurrent_kernel* 
_output_shapes
:
*
dtype0

gru_1/gru_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namegru_1/gru_cell_3/kernel

+gru_1/gru_cell_3/kernel/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_3/kernel* 
_output_shapes
:
*
dtype0

gru/gru_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_namegru/gru_cell_2/bias
|
'gru/gru_cell_2/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell_2/bias*
_output_shapes
:	*
dtype0

gru/gru_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!gru/gru_cell_2/recurrent_kernel

3gru/gru_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell_2/recurrent_kernel* 
_output_shapes
:
*
dtype0

gru/gru_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_namegru/gru_cell_2/kernel

)gru/gru_cell_2/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell_2/kernel*
_output_shapes
:	*
dtype0
}
dense_surface/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namedense_surface/bias
v
&dense_surface/bias/Read/ReadVariableOpReadVariableOpdense_surface/bias*
_output_shapes	
:*
dtype0

dense_surface/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_namedense_surface/kernel

(dense_surface/kernel/Read/ReadVariableOpReadVariableOpdense_surface/kernel* 
_output_shapes
:
*
dtype0
}
serving_default_inputs_auxPlaceholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

serving_default_inputs_mainPlaceholder*+
_output_shapes
:’’’’’’’’’1*
dtype0* 
shape:’’’’’’’’’1
ą
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputs_auxserving_default_inputs_maingru/gru_cell_2/biasgru/gru_cell_2/kernelgru/gru_cell_2/recurrent_kerneldense_surface/kerneldense_surface/biasgru_1/gru_cell_3/biasgru_1/gru_cell_3/kernel!gru_1/gru_cell_3/recurrent_kerneldense_output/kerneldense_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_388394

NoOpNoOp
£>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ž=
valueŌ=BŃ= BŹ=
Ą
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
Į
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
¦
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
Į
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

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
°
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


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
Ó
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

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


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
Ų
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

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

0
1
2*
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
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

 non_trainable_variables
”layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

„trace_0
¦trace_1* 

§trace_0
Øtrace_1* 
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

©non_trainable_variables
Ŗlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

®trace_0* 

Ætrace_0* 
<
°	variables
±	keras_api

²total

³count*
M
“	variables
µ	keras_api

¶total

·count
ø
_fn_kwargs*
M
¹	variables
ŗ	keras_api

»total

¼count
½
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
²0
³1*

°	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

¶0
·1*

“	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

»0
¼1*

¹	variables*
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
ī
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_390890
é
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_390954ŪÓ 

Ä
H__inference_dense_output_layer_call_and_return_conditional_losses_387381

inputs
dense_387371:	
dense_387373:
identity¢dense/StatefulPartitionedCallI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
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
valueB"’’’’   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’ń
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_387371dense_387373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_387350\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’’’’’’’’’’: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
N

?__inference_gru_layer_call_and_return_conditional_losses_387929

inputs5
"gru_cell_2_readvariableop_resource:	<
)gru_cell_2_matmul_readvariableop_resource:	?
+gru_cell_2_matmul_1_readvariableop_resource:

identity

identity_1¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:1’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_387839*
condR
while_cond_387838*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ć
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:1’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’1[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’1`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:’’’’’’’’’²
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’1: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’1
 
_user_specified_nameinputs
³=

while_body_390065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_3_readvariableop_resource_0:	E
1while_gru_cell_3_matmul_readvariableop_resource_0:
G
3while_gru_cell_3_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_3_readvariableop_resource:	C
/while_gru_cell_3_matmul_readvariableop_resource:
E
1while_gru_cell_3_matmul_1_readvariableop_resource:
¢&while/gru_cell_3/MatMul/ReadVariableOp¢(while/gru_cell_3/MatMul_1/ReadVariableOp¢while/gru_cell_3/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
Ģ

&__inference_model_layer_call_fn_388254
inputs_main

inputs_aux
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:

	unknown_7:	
	unknown_8:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputs_main
inputs_auxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_388231s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:’’’’’’’’’1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
inputs_aux
¾5

?__inference_gru_layer_call_and_return_conditional_losses_386787

inputs$
gru_cell_2_386710:	$
gru_cell_2_386712:	%
gru_cell_2_386714:

identity

identity_1¢"gru_cell_2/StatefulPartitionedCall¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_maskÉ
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_386710gru_cell_2_386712gru_cell_2_386714*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_386709n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_386710gru_cell_2_386712gru_cell_2_386714*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_386722*
condR
while_cond_386721*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ģ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:’’’’’’’’’s
NoOpNoOp#^gru_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
²
»
&__inference_gru_1_layer_call_fn_389811
inputs_0
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_387132}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
Ņ 
²
while_body_387068
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_3_387090_0:	-
while_gru_cell_3_387092_0:
-
while_gru_cell_3_387094_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_3_387090:	+
while_gru_cell_3_387092:
+
while_gru_cell_3_387094:
¢(while/gru_cell_3/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
(while/gru_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_3_387090_0while_gru_cell_3_387092_0while_gru_cell_3_387094_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_387055Ś
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity1while/gru_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w

while/NoOpNoOp)^while/gru_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_3_387090while_gru_cell_3_387090_0"4
while_gru_cell_3_387092while_gru_cell_3_387092_0"4
while_gru_cell_3_387094while_gru_cell_3_387094_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2T
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
ĪN

?__inference_gru_layer_call_and_return_conditional_losses_389318
inputs_05
"gru_cell_2_readvariableop_resource:	<
)gru_cell_2_matmul_readvariableop_resource:	?
+gru_cell_2_matmul_1_readvariableop_resource:

identity

identity_1¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_389228*
condR
while_cond_389227*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ģ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:’’’’’’’’’²
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
½	
Ę
$__inference_gru_layer_call_fn_389164

inputs
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1¢StatefulPartitionedCallž
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:’’’’’’’’’1:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_387929t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’1r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’1
 
_user_specified_nameinputs
Ļ 
°
while_body_386722
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_2_386744_0:	,
while_gru_cell_2_386746_0:	-
while_gru_cell_2_386748_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_2_386744:	*
while_gru_cell_2_386746:	+
while_gru_cell_2_386748:
¢(while/gru_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_386744_0while_gru_cell_2_386746_0while_gru_cell_2_386748_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_386709Ś
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w

while/NoOpNoOp)^while/gru_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_2_386744while_gru_cell_2_386744_0"4
while_gru_cell_2_386746while_gru_cell_2_386746_0"4
while_gru_cell_2_386748while_gru_cell_2_386748_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2T
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
²	
ö
gru_while_cond_388510$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1<
8gru_while_gru_while_cond_388510___redundant_placeholder0<
8gru_while_gru_while_cond_388510___redundant_placeholder1<
8gru_while_gru_while_cond_388510___redundant_placeholder2<
8gru_while_gru_while_cond_388510___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::N J
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
ē	
Č
$__inference_gru_layer_call_fn_389138
inputs_0
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:’’’’’’’’’’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_386930}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
¾
Ŗ
while_cond_389909
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_389909___redundant_placeholder04
0while_while_cond_389909___redundant_placeholder14
0while_while_cond_389909___redundant_placeholder24
0while_while_cond_389909___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
Æ=

while_body_387839
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_2_readvariableop_resource_0:	D
1while_gru_cell_2_matmul_readvariableop_resource_0:	G
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_2_readvariableop_resource:	B
/while_gru_cell_2_matmul_readvariableop_resource:	E
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¶
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
Ģ

&__inference_model_layer_call_fn_388188
inputs_main

inputs_aux
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:

	unknown_7:	
	unknown_8:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputs_main
inputs_auxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_388165s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:’’’’’’’’’1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
inputs_aux
°

H__inference_dense_output_layer_call_and_return_conditional_losses_390526

inputs7
$dense_matmul_readvariableop_resource:	3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
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
valueB"’’’’   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense/Sigmoid:y:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’’’’’’’’’’: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ŁN

A__inference_gru_1_layer_call_and_return_conditional_losses_390464

inputs5
"gru_cell_3_readvariableop_resource:	=
)gru_cell_3_matmul_readvariableop_resource:
?
+gru_cell_3_matmul_1_readvariableop_resource:

identity¢ gru_cell_3/MatMul/ReadVariableOp¢"gru_cell_3/MatMul_1/ReadVariableOp¢gru_cell_3/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: u
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*,
_output_shapes
:2’’’’’’’’’
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:ź
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390375*
condR
while_cond_390374*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ć
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’2²
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:’’’’’’’’’2: : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
6

A__inference_gru_1_layer_call_and_return_conditional_losses_387132

inputs$
gru_cell_3_387056:	%
gru_cell_3_387058:
%
gru_cell_3_387060:

identity¢"gru_cell_3/StatefulPartitionedCall¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ~
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:ź
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskÉ
"gru_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_3_387056gru_cell_3_387058gru_cell_3_387060*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_387055n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_3_387056gru_cell_3_387058gru_cell_3_387060*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_387068*
condR
while_cond_387067*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ģ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’s
NoOpNoOp#^gru_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’’’’’’’’’’: : : 2H
"gru_cell_3/StatefulPartitionedCall"gru_cell_3/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
øB
ł
gru_while_body_388844$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0A
.gru_while_gru_cell_2_readvariableop_resource_0:	H
5gru_while_gru_cell_2_matmul_readvariableop_resource_0:	K
7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:

gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor?
,gru_while_gru_cell_2_readvariableop_resource:	F
3gru_while_gru_cell_2_matmul_readvariableop_resource:	I
5gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢*gru/while/gru_cell_2/MatMul/ReadVariableOp¢,gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢#gru/while/gru_cell_2/ReadVariableOp
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ŗ
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
#gru/while/gru_cell_2/ReadVariableOpReadVariableOp.gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru/while/gru_cell_2/unstackUnpack+gru/while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num”
*gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp5gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ā
gru/while/gru_cell_2/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:02gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
gru/while/gru_cell_2/BiasAddBiasAdd%gru/while/gru_cell_2/MatMul:product:0%gru/while/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’o
$gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’å
gru/while/gru_cell_2/splitSplit-gru/while/gru_cell_2/split/split_dim:output:0%gru/while/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split¦
,gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0©
gru/while/gru_cell_2/MatMul_1MatMulgru_while_placeholder_24gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¬
gru/while/gru_cell_2/BiasAdd_1BiasAdd'gru/while/gru_cell_2/MatMul_1:product:0%gru/while/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’o
gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’q
&gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
gru/while/gru_cell_2/split_1SplitV'gru/while/gru_cell_2/BiasAdd_1:output:0#gru/while/gru_cell_2/Const:output:0/gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split 
gru/while/gru_cell_2/addAddV2#gru/while/gru_cell_2/split:output:0%gru/while/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru/while/gru_cell_2/SigmoidSigmoidgru/while/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’¢
gru/while/gru_cell_2/add_1AddV2#gru/while/gru_cell_2/split:output:1%gru/while/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’|
gru/while/gru_cell_2/Sigmoid_1Sigmoidgru/while/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/mulMul"gru/while/gru_cell_2/Sigmoid_1:y:0%gru/while/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/add_2AddV2#gru/while/gru_cell_2/split:output:2gru/while/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
gru/while/gru_cell_2/ReluRelugru/while/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/mul_1Mul gru/while/gru_cell_2/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’_
gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/while/gru_cell_2/subSub#gru/while/gru_cell_2/sub/x:output:0 gru/while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/mul_2Mulgru/while/gru_cell_2/sub:z:0'gru/while/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/add_3AddV2gru/while/gru_cell_2/mul_1:z:0gru/while/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ó
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅQ
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
: 
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: 
gru/while/Identity_4Identitygru/while/gru_cell_2/add_3:z:0^gru/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ņ
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
gru_while_identity_4gru/while/Identity_4:output:0"ø
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2X
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
N

?__inference_gru_layer_call_and_return_conditional_losses_389626

inputs5
"gru_cell_2_readvariableop_resource:	<
)gru_cell_2_matmul_readvariableop_resource:	?
+gru_cell_2_matmul_1_readvariableop_resource:

identity

identity_1¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:1’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_389536*
condR
while_cond_389535*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ć
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:1’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’1[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’1`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:’’’’’’’’’²
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’1: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’1
 
_user_specified_nameinputs
Ö

.__inference_dense_surface_layer_call_fn_389789

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallā
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_387590p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ą#
Ų
A__inference_model_layer_call_and_return_conditional_losses_388165

inputs
inputs_1

gru_388129:	

gru_388131:	

gru_388133:
(
dense_surface_388139:
#
dense_surface_388141:	
gru_1_388148:	 
gru_1_388150:
 
gru_1_388152:
&
dense_output_388157:	!
dense_output_388159:
identity¢$dense_output/StatefulPartitionedCall¢%dense_surface/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCall
gru/StatefulPartitionedCallStatefulPartitionedCallinputs
gru_388129
gru_388131
gru_388133*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:’’’’’’’’’1:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_387568Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¬
tf.concat_3/concatConcatV2$gru/StatefulPartitionedCall:output:1inputs_1 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
%dense_surface/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_surface_388139dense_surface_388141*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_387590o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’      «
tf.reshape_1/ReshapeReshape.dense_surface/StatefulPartitionedCall:output:0#tf.reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
tf.concat_4/concatConcatV2$gru/StatefulPartitionedCall:output:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2
gru_1/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0gru_1_388148gru_1_388150gru_1_388152*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:’’’’’’’’’2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_387754Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0&gru_1/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2
$dense_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_output_388157dense_output_388159*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_387361k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’
IdentityIdentity-dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2Ó
NoOpNoOp%^dense_output/StatefulPartitionedCall&^dense_surface/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall2N
%dense_surface/StatefulPartitionedCall%dense_surface/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’1
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
 O

A__inference_gru_1_layer_call_and_return_conditional_losses_390154
inputs_05
"gru_cell_3_readvariableop_resource:	=
)gru_cell_3_matmul_readvariableop_resource:
?
+gru_cell_3_matmul_1_readvariableop_resource:

identity¢ gru_cell_3/MatMul/ReadVariableOp¢"gru_cell_3/MatMul_1/ReadVariableOp¢gru_cell_3/ReadVariableOp¢whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ~
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:ź
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390065*
condR
while_cond_390064*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ģ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’²
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’’’’’’’’’’: : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
Ņ 
²
while_body_387212
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_3_387234_0:	-
while_gru_cell_3_387236_0:
-
while_gru_cell_3_387238_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_3_387234:	+
while_gru_cell_3_387236:
+
while_gru_cell_3_387238:
¢(while/gru_cell_3/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
(while/gru_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_3_387234_0while_gru_cell_3_387236_0while_gru_cell_3_387238_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_387199Ś
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity1while/gru_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w

while/NoOpNoOp)^while/gru_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_3_387234while_gru_cell_3_387234_0"4
while_gru_cell_3_387236while_gru_cell_3_387236_0"4
while_gru_cell_3_387238while_gru_cell_3_387238_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2T
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
¾
Ŗ
while_cond_388013
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_388013___redundant_placeholder04
0while_while_cond_388013___redundant_placeholder14
0while_while_cond_388013___redundant_placeholder24
0while_while_cond_388013___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
ŁN

A__inference_gru_1_layer_call_and_return_conditional_losses_390309

inputs5
"gru_cell_3_readvariableop_resource:	=
)gru_cell_3_matmul_readvariableop_resource:
?
+gru_cell_3_matmul_1_readvariableop_resource:

identity¢ gru_cell_3/MatMul/ReadVariableOp¢"gru_cell_3/MatMul_1/ReadVariableOp¢gru_cell_3/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: u
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*,
_output_shapes
:2’’’’’’’’’
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:ź
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390220*
condR
while_cond_390219*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ć
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’2²
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:’’’’’’’’’2: : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
³=

while_body_390220
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_3_readvariableop_resource_0:	E
1while_gru_cell_3_matmul_readvariableop_resource_0:
G
3while_gru_cell_3_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_3_readvariableop_resource:	C
/while_gru_cell_3_matmul_readvariableop_resource:
E
1while_gru_cell_3_matmul_1_readvariableop_resource:
¢&while/gru_cell_3/MatMul/ReadVariableOp¢(while/gru_cell_3/MatMul_1/ReadVariableOp¢while/gru_cell_3/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
²	
ö
gru_while_cond_388843$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1<
8gru_while_gru_while_cond_388843___redundant_placeholder0<
8gru_while_gru_while_cond_388843___redundant_placeholder1<
8gru_while_gru_while_cond_388843___redundant_placeholder2<
8gru_while_gru_while_cond_388843___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::N J
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
¬
Ü
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_390738

inputs
states_0*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’É
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:’’’’’’’’’J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’:’’’’’’’’’: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0
¤
Ś
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_387055

inputs

states*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’É
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:’’’’’’’’’J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’:’’’’’’’’’: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:PL
(
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates
J
­

model_gru_while_body_3863740
,model_gru_while_model_gru_while_loop_counter6
2model_gru_while_model_gru_while_maximum_iterations
model_gru_while_placeholder!
model_gru_while_placeholder_1!
model_gru_while_placeholder_2/
+model_gru_while_model_gru_strided_slice_1_0k
gmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0G
4model_gru_while_gru_cell_2_readvariableop_resource_0:	N
;model_gru_while_gru_cell_2_matmul_readvariableop_resource_0:	Q
=model_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:

model_gru_while_identity
model_gru_while_identity_1
model_gru_while_identity_2
model_gru_while_identity_3
model_gru_while_identity_4-
)model_gru_while_model_gru_strided_slice_1i
emodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensorE
2model_gru_while_gru_cell_2_readvariableop_resource:	L
9model_gru_while_gru_cell_2_matmul_readvariableop_resource:	O
;model_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢0model/gru/while/gru_cell_2/MatMul/ReadVariableOp¢2model/gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢)model/gru/while/gru_cell_2/ReadVariableOp
Amodel/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ų
3model/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemgmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0model_gru_while_placeholderJmodel/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
)model/gru/while/gru_cell_2/ReadVariableOpReadVariableOp4model_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	*
dtype0
"model/gru/while/gru_cell_2/unstackUnpack1model/gru/while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num­
0model/gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp;model_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ō
!model/gru/while/gru_cell_2/MatMulMatMul:model/gru/while/TensorArrayV2Read/TensorListGetItem:item:08model/gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ŗ
"model/gru/while/gru_cell_2/BiasAddBiasAdd+model/gru/while/gru_cell_2/MatMul:product:0+model/gru/while/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
*model/gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’÷
 model/gru/while/gru_cell_2/splitSplit3model/gru/while/gru_cell_2/split/split_dim:output:0+model/gru/while/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split²
2model/gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp=model_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0»
#model/gru/while/gru_cell_2/MatMul_1MatMulmodel_gru_while_placeholder_2:model/gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¾
$model/gru/while/gru_cell_2/BiasAdd_1BiasAdd-model/gru/while/gru_cell_2/MatMul_1:product:0+model/gru/while/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’u
 model/gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’w
,model/gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’µ
"model/gru/while/gru_cell_2/split_1SplitV-model/gru/while/gru_cell_2/BiasAdd_1:output:0)model/gru/while/gru_cell_2/Const:output:05model/gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split²
model/gru/while/gru_cell_2/addAddV2)model/gru/while/gru_cell_2/split:output:0+model/gru/while/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
"model/gru/while/gru_cell_2/SigmoidSigmoid"model/gru/while/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’“
 model/gru/while/gru_cell_2/add_1AddV2)model/gru/while/gru_cell_2/split:output:1+model/gru/while/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
$model/gru/while/gru_cell_2/Sigmoid_1Sigmoid$model/gru/while/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’Æ
model/gru/while/gru_cell_2/mulMul(model/gru/while/gru_cell_2/Sigmoid_1:y:0+model/gru/while/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’«
 model/gru/while/gru_cell_2/add_2AddV2)model/gru/while/gru_cell_2/split:output:2"model/gru/while/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/gru/while/gru_cell_2/ReluRelu$model/gru/while/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’”
 model/gru/while/gru_cell_2/mul_1Mul&model/gru/while/gru_cell_2/Sigmoid:y:0model_gru_while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’e
 model/gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?«
model/gru/while/gru_cell_2/subSub)model/gru/while/gru_cell_2/sub/x:output:0&model/gru/while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’­
 model/gru/while/gru_cell_2/mul_2Mul"model/gru/while/gru_cell_2/sub:z:0-model/gru/while/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
 model/gru/while/gru_cell_2/add_3AddV2$model/gru/while/gru_cell_2/mul_1:z:0$model/gru/while/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’ė
4model/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmodel_gru_while_placeholder_1model_gru_while_placeholder$model/gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅW
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
value	B :
model/gru/while/add_1AddV2,model_gru_while_model_gru_while_loop_counter model/gru/while/add_1/y:output:0*
T0*
_output_shapes
: w
model/gru/while/IdentityIdentitymodel/gru/while/add_1:z:0^model/gru/while/NoOp*
T0*
_output_shapes
: 
model/gru/while/Identity_1Identity2model_gru_while_model_gru_while_maximum_iterations^model/gru/while/NoOp*
T0*
_output_shapes
: w
model/gru/while/Identity_2Identitymodel/gru/while/add:z:0^model/gru/while/NoOp*
T0*
_output_shapes
: ¤
model/gru/while/Identity_3IdentityDmodel/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/gru/while/NoOp*
T0*
_output_shapes
: 
model/gru/while/Identity_4Identity$model/gru/while/gru_cell_2/add_3:z:0^model/gru/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’ź
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
)model_gru_while_model_gru_strided_slice_1+model_gru_while_model_gru_strided_slice_1_0"Š
emodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensorgmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2d
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
¾5

?__inference_gru_layer_call_and_return_conditional_losses_386930

inputs$
gru_cell_2_386853:	$
gru_cell_2_386855:	%
gru_cell_2_386857:

identity

identity_1¢"gru_cell_2/StatefulPartitionedCall¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_maskÉ
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_386853gru_cell_2_386855gru_cell_2_386857*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_386852n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_386853gru_cell_2_386855gru_cell_2_386857*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_386865*
condR
while_cond_386864*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ģ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:’’’’’’’’’s
NoOpNoOp#^gru_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
÷#
ß
A__inference_model_layer_call_and_return_conditional_losses_388121
inputs_main

inputs_aux

gru_387930:	

gru_387932:	

gru_387934:
(
dense_surface_387940:
#
dense_surface_387942:	
gru_1_388104:	 
gru_1_388106:
 
gru_1_388108:
&
dense_output_388113:	!
dense_output_388115:
identity¢$dense_output/StatefulPartitionedCall¢%dense_surface/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCall
gru/StatefulPartitionedCallStatefulPartitionedCallinputs_main
gru_387930
gru_387932
gru_387934*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:’’’’’’’’’1:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_387929Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :®
tf.concat_3/concatConcatV2$gru/StatefulPartitionedCall:output:1
inputs_aux tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
%dense_surface/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_surface_387940dense_surface_387942*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_387590o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’      «
tf.reshape_1/ReshapeReshape.dense_surface/StatefulPartitionedCall:output:0#tf.reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
tf.concat_4/concatConcatV2$gru/StatefulPartitionedCall:output:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2
gru_1/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0gru_1_388104gru_1_388106gru_1_388108*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:’’’’’’’’’2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_388103Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0&gru_1/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2
$dense_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_output_388113dense_output_388115*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_387381k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’
IdentityIdentity-dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2Ó
NoOpNoOp%^dense_output/StatefulPartitionedCall&^dense_surface/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall2N
%dense_surface/StatefulPartitionedCall%dense_surface/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:X T
+
_output_shapes
:’’’’’’’’’1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
inputs_aux
N
ģ

"__inference__traced_restore_390954
file_prefix9
%assignvariableop_dense_surface_kernel:
4
%assignvariableop_1_dense_surface_bias:	;
(assignvariableop_2_gru_gru_cell_2_kernel:	F
2assignvariableop_3_gru_gru_cell_2_recurrent_kernel:
9
&assignvariableop_4_gru_gru_cell_2_bias:	>
*assignvariableop_5_gru_1_gru_cell_3_kernel:
H
4assignvariableop_6_gru_1_gru_cell_3_recurrent_kernel:
;
(assignvariableop_7_gru_1_gru_cell_3_bias:	9
&assignvariableop_8_dense_output_kernel:	2
$assignvariableop_9_dense_output_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: %
assignvariableop_12_total_2: %
assignvariableop_13_count_2: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ę
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB’B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ż
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ø
AssignVariableOpAssignVariableOp%assignvariableop_dense_surface_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_1AssignVariableOp%assignvariableop_1_dense_surface_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:æ
AssignVariableOp_2AssignVariableOp(assignvariableop_2_gru_gru_cell_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_3AssignVariableOp2assignvariableop_3_gru_gru_cell_2_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_4AssignVariableOp&assignvariableop_4_gru_gru_cell_2_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Į
AssignVariableOp_5AssignVariableOp*assignvariableop_5_gru_1_gru_cell_3_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ė
AssignVariableOp_6AssignVariableOp4assignvariableop_6_gru_1_gru_cell_3_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:æ
AssignVariableOp_7AssignVariableOp(assignvariableop_7_gru_1_gru_cell_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_8AssignVariableOp&assignvariableop_8_dense_output_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_9AssignVariableOp$assignvariableop_9_dense_output_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:¶
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ŗ
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_2Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_2Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Ū
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: Č
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
¾
Ŗ
while_cond_386721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_386721___redundant_placeholder04
0while_while_cond_386721___redundant_placeholder14
0while_while_cond_386721___redundant_placeholder24
0while_while_cond_386721___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
ŁN

A__inference_gru_1_layer_call_and_return_conditional_losses_388103

inputs5
"gru_cell_3_readvariableop_resource:	=
)gru_cell_3_matmul_readvariableop_resource:
?
+gru_cell_3_matmul_1_readvariableop_resource:

identity¢ gru_cell_3/MatMul/ReadVariableOp¢"gru_cell_3/MatMul_1/ReadVariableOp¢gru_cell_3/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: u
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*,
_output_shapes
:2’’’’’’’’’
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:ź
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_388014*
condR
while_cond_388013*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ć
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’2²
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:’’’’’’’’’2: : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs


model_gru_1_while_cond_3865374
0model_gru_1_while_model_gru_1_while_loop_counter:
6model_gru_1_while_model_gru_1_while_maximum_iterations!
model_gru_1_while_placeholder#
model_gru_1_while_placeholder_1#
model_gru_1_while_placeholder_26
2model_gru_1_while_less_model_gru_1_strided_slice_1L
Hmodel_gru_1_while_model_gru_1_while_cond_386537___redundant_placeholder0L
Hmodel_gru_1_while_model_gru_1_while_cond_386537___redundant_placeholder1L
Hmodel_gru_1_while_model_gru_1_while_cond_386537___redundant_placeholder2L
Hmodel_gru_1_while_model_gru_1_while_cond_386537___redundant_placeholder3
model_gru_1_while_identity

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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::V R
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:


-__inference_dense_output_layer_call_fn_390473

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallķ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_387361|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
½	
Ę
$__inference_gru_layer_call_fn_389151

inputs
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1¢StatefulPartitionedCallž
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:’’’’’’’’’1:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_387568t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’1r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’1
 
_user_specified_nameinputs
¾
Ŗ
while_cond_387838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_387838___redundant_placeholder04
0while_while_cond_387838___redundant_placeholder14
0while_while_cond_387838___redundant_placeholder24
0while_while_cond_387838___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:


ó
A__inference_dense_layer_call_and_return_conditional_losses_390758

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
N

?__inference_gru_layer_call_and_return_conditional_losses_389780

inputs5
"gru_cell_2_readvariableop_resource:	<
)gru_cell_2_matmul_readvariableop_resource:	?
+gru_cell_2_matmul_1_readvariableop_resource:

identity

identity_1¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:1’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_389690*
condR
while_cond_389689*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ć
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:1’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’1[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’1`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:’’’’’’’’’²
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’1: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’1
 
_user_specified_nameinputs
¾

__inference__traced_save_390890
file_prefix?
+read_disablecopyonread_dense_surface_kernel:
:
+read_1_disablecopyonread_dense_surface_bias:	A
.read_2_disablecopyonread_gru_gru_cell_2_kernel:	L
8read_3_disablecopyonread_gru_gru_cell_2_recurrent_kernel:
?
,read_4_disablecopyonread_gru_gru_cell_2_bias:	D
0read_5_disablecopyonread_gru_1_gru_cell_3_kernel:
N
:read_6_disablecopyonread_gru_1_gru_cell_3_recurrent_kernel:
A
.read_7_disablecopyonread_gru_1_gru_cell_3_bias:	?
,read_8_disablecopyonread_dense_output_kernel:	8
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
identity_37¢MergeV2Checkpoints¢Read/DisableCopyOnRead¢Read/ReadVariableOp¢Read_1/DisableCopyOnRead¢Read_1/ReadVariableOp¢Read_10/DisableCopyOnRead¢Read_10/ReadVariableOp¢Read_11/DisableCopyOnRead¢Read_11/ReadVariableOp¢Read_12/DisableCopyOnRead¢Read_12/ReadVariableOp¢Read_13/DisableCopyOnRead¢Read_13/ReadVariableOp¢Read_14/DisableCopyOnRead¢Read_14/ReadVariableOp¢Read_15/DisableCopyOnRead¢Read_15/ReadVariableOp¢Read_16/DisableCopyOnRead¢Read_16/ReadVariableOp¢Read_17/DisableCopyOnRead¢Read_17/ReadVariableOp¢Read_2/DisableCopyOnRead¢Read_2/ReadVariableOp¢Read_3/DisableCopyOnRead¢Read_3/ReadVariableOp¢Read_4/DisableCopyOnRead¢Read_4/ReadVariableOp¢Read_5/DisableCopyOnRead¢Read_5/ReadVariableOp¢Read_6/DisableCopyOnRead¢Read_6/ReadVariableOp¢Read_7/DisableCopyOnRead¢Read_7/ReadVariableOp¢Read_8/DisableCopyOnRead¢Read_8/ReadVariableOp¢Read_9/DisableCopyOnRead¢Read_9/ReadVariableOpw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: }
Read/DisableCopyOnReadDisableCopyOnRead+read_disablecopyonread_dense_surface_kernel"/device:CPU:0*
_output_shapes
 ©
Read/ReadVariableOpReadVariableOp+read_disablecopyonread_dense_surface_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_1/DisableCopyOnReadDisableCopyOnRead+read_1_disablecopyonread_dense_surface_bias"/device:CPU:0*
_output_shapes
 Ø
Read_1/ReadVariableOpReadVariableOp+read_1_disablecopyonread_dense_surface_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_2/DisableCopyOnReadDisableCopyOnRead.read_2_disablecopyonread_gru_gru_cell_2_kernel"/device:CPU:0*
_output_shapes
 Æ
Read_2/ReadVariableOpReadVariableOp.read_2_disablecopyonread_gru_gru_cell_2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_3/DisableCopyOnReadDisableCopyOnRead8read_3_disablecopyonread_gru_gru_cell_2_recurrent_kernel"/device:CPU:0*
_output_shapes
 ŗ
Read_3/ReadVariableOpReadVariableOp8read_3_disablecopyonread_gru_gru_cell_2_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
e

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_4/DisableCopyOnReadDisableCopyOnRead,read_4_disablecopyonread_gru_gru_cell_2_bias"/device:CPU:0*
_output_shapes
 ­
Read_4/ReadVariableOpReadVariableOp,read_4_disablecopyonread_gru_gru_cell_2_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_5/DisableCopyOnReadDisableCopyOnRead0read_5_disablecopyonread_gru_1_gru_cell_3_kernel"/device:CPU:0*
_output_shapes
 ²
Read_5/ReadVariableOpReadVariableOp0read_5_disablecopyonread_gru_1_gru_cell_3_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0p
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_6/DisableCopyOnReadDisableCopyOnRead:read_6_disablecopyonread_gru_1_gru_cell_3_recurrent_kernel"/device:CPU:0*
_output_shapes
 ¼
Read_6/ReadVariableOpReadVariableOp:read_6_disablecopyonread_gru_1_gru_cell_3_recurrent_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_7/DisableCopyOnReadDisableCopyOnRead.read_7_disablecopyonread_gru_1_gru_cell_3_bias"/device:CPU:0*
_output_shapes
 Æ
Read_7/ReadVariableOpReadVariableOp.read_7_disablecopyonread_gru_1_gru_cell_3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_8/DisableCopyOnReadDisableCopyOnRead,read_8_disablecopyonread_dense_output_kernel"/device:CPU:0*
_output_shapes
 ­
Read_8/ReadVariableOpReadVariableOp,read_8_disablecopyonread_dense_output_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	~
Read_9/DisableCopyOnReadDisableCopyOnRead*read_9_disablecopyonread_dense_output_bias"/device:CPU:0*
_output_shapes
 ¦
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
 
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
 ”
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
 
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
 
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
 
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
 
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
 
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
 
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
: ć
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB’B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ń
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
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
: ż
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
æ

Ü
+__inference_gru_cell_3_layer_call_fn_390660

inputs
states_0
unknown:	
	unknown_0:

	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_387199p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’:’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0
Č
§	
A__inference_model_layer_call_and_return_conditional_losses_388779
inputs_0
inputs_19
&gru_gru_cell_2_readvariableop_resource:	@
-gru_gru_cell_2_matmul_readvariableop_resource:	C
/gru_gru_cell_2_matmul_1_readvariableop_resource:
@
,dense_surface_matmul_readvariableop_resource:
<
-dense_surface_biasadd_readvariableop_resource:	;
(gru_1_gru_cell_3_readvariableop_resource:	C
/gru_1_gru_cell_3_matmul_readvariableop_resource:
E
1gru_1_gru_cell_3_matmul_1_readvariableop_resource:
D
1dense_output_dense_matmul_readvariableop_resource:	@
2dense_output_dense_biasadd_readvariableop_resource:
identity¢)dense_output/dense/BiasAdd/ReadVariableOp¢(dense_output/dense/MatMul/ReadVariableOp¢$dense_surface/BiasAdd/ReadVariableOp¢#dense_surface/MatMul/ReadVariableOp¢$gru/gru_cell_2/MatMul/ReadVariableOp¢&gru/gru_cell_2/MatMul_1/ReadVariableOp¢gru/gru_cell_2/ReadVariableOp¢	gru/while¢&gru_1/gru_cell_3/MatMul/ReadVariableOp¢(gru_1/gru_cell_3/MatMul_1/ReadVariableOp¢gru_1/gru_cell_3/ReadVariableOp¢gru_1/whileO
	gru/ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻa
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
valueB:å
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
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
 *    y
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
gru/transpose	Transposeinputs_0gru/transpose/perm:output:0*
T0*+
_output_shapes
:1’’’’’’’’’Z
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
::ķĻc
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
valueB:ļ
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
’’’’’’’’’Ą
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ģ
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅc
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
valueB:ż
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask
gru/gru_cell_2/ReadVariableOpReadVariableOp&gru_gru_cell_2_readvariableop_resource*
_output_shapes
:	*
dtype0
gru/gru_cell_2/unstackUnpack%gru/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp-gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru/gru_cell_2/MatMulMatMulgru/strided_slice_2:output:0,gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/BiasAddBiasAddgru/gru_cell_2/MatMul:product:0gru/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ó
gru/gru_cell_2/splitSplit'gru/gru_cell_2/split/split_dim:output:0gru/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
&gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp/gru_gru_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru/gru_cell_2/MatMul_1MatMulgru/zeros:output:0.gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/BiasAdd_1BiasAdd!gru/gru_cell_2/MatMul_1:product:0gru/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’i
gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’k
 gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
gru/gru_cell_2/split_1SplitV!gru/gru_cell_2/BiasAdd_1:output:0gru/gru_cell_2/Const:output:0)gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru/gru_cell_2/addAddV2gru/gru_cell_2/split:output:0gru/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’l
gru/gru_cell_2/SigmoidSigmoidgru/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/add_1AddV2gru/gru_cell_2/split:output:1gru/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’p
gru/gru_cell_2/Sigmoid_1Sigmoidgru/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/mulMulgru/gru_cell_2/Sigmoid_1:y:0gru/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/add_2AddV2gru/gru_cell_2/split:output:2gru/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’h
gru/gru_cell_2/ReluRelugru/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
gru/gru_cell_2/mul_1Mulgru/gru_cell_2/Sigmoid:y:0gru/zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/gru_cell_2/subSubgru/gru_cell_2/sub/x:output:0gru/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/mul_2Mulgru/gru_cell_2/sub:z:0!gru/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/add_3AddV2gru/gru_cell_2/mul_1:z:0gru/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ä
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅJ
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
’’’’’’’’’X
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ń
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0&gru_gru_cell_2_readvariableop_resource-gru_gru_cell_2_matmul_readvariableop_resource/gru_gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *!
bodyR
gru_while_body_388511*!
condR
gru_while_cond_388510*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ļ
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:1’’’’’’’’’*
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’e
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’1_
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
value	B :
tf.concat_3/concatConcatV2gru/while:output:4inputs_1 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
#dense_surface/MatMul/ReadVariableOpReadVariableOp,dense_surface_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_surface/MatMulMatMultf.concat_3/concat:output:0+dense_surface/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
$dense_surface/BiasAdd/ReadVariableOpReadVariableOp-dense_surface_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0”
dense_surface/BiasAddBiasAdddense_surface/MatMul:product:0,dense_surface/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
dense_surface/ReluReludense_surface/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’      
tf.reshape_1/ReshapeReshape dense_surface/Relu:activations:0#tf.reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :“
tf.concat_4/concatConcatV2gru/transpose_1:y:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2d
gru_1/ShapeShapetf.concat_4/concat:output:0*
T0*
_output_shapes
::ķĻc
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
valueB:ļ
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
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
 *    
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_1/transpose	Transposetf.concat_4/concat:output:0gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:2’’’’’’’’’^
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
::ķĻe
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
valueB:ł
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
’’’’’’’’’Ę
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ^
gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
gru_1/ReverseV2	ReverseV2gru_1/transpose:y:0gru_1/ReverseV2/axis:output:0*
T0*,
_output_shapes
:2’’’’’’’’’
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ÷
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/ReverseV2:output:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅe
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
valueB:
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask
gru_1/gru_cell_3/ReadVariableOpReadVariableOp(gru_1_gru_cell_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_1/gru_cell_3/unstackUnpack'gru_1/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&gru_1/gru_cell_3/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¤
gru_1/gru_cell_3/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/BiasAddBiasAdd!gru_1/gru_cell_3/MatMul:product:0!gru_1/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 gru_1/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
gru_1/gru_cell_3/splitSplit)gru_1/gru_cell_3/split/split_dim:output:0!gru_1/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(gru_1/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_1/gru_cell_3/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
gru_1/gru_cell_3/BiasAdd_1BiasAdd#gru_1/gru_cell_3/MatMul_1:product:0!gru_1/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
gru_1/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"gru_1/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
gru_1/gru_cell_3/split_1SplitV#gru_1/gru_cell_3/BiasAdd_1:output:0gru_1/gru_cell_3/Const:output:0+gru_1/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_1/gru_cell_3/addAddV2gru_1/gru_cell_3/split:output:0!gru_1/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
gru_1/gru_cell_3/SigmoidSigmoidgru_1/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/add_1AddV2gru_1/gru_cell_3/split:output:1!gru_1/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
gru_1/gru_cell_3/Sigmoid_1Sigmoidgru_1/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/mulMulgru_1/gru_cell_3/Sigmoid_1:y:0!gru_1/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/add_2AddV2gru_1/gru_cell_3/split:output:2gru_1/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
gru_1/gru_cell_3/ReluRelugru_1/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/mul_1Mulgru_1/gru_cell_3/Sigmoid:y:0gru_1/zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’[
gru_1/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/gru_cell_3/subSubgru_1/gru_cell_3/sub/x:output:0gru_1/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/mul_2Mulgru_1/gru_cell_3/sub:z:0#gru_1/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/add_3AddV2gru_1/gru_cell_3/mul_1:z:0gru_1/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ź
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅL

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
’’’’’’’’’Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_3_readvariableop_resource/gru_1_gru_cell_3_matmul_readvariableop_resource1gru_1_gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_1_while_body_388675*#
condR
gru_1_while_cond_388674*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Õ
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2’’’’’’’’’*
element_dtype0n
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’g
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2a
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
value	B :“
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0gru_1/transpose_1:y:0 tf.concat_5/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’
(dense_output/dense/MatMul/ReadVariableOpReadVariableOp1dense_output_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¦
dense_output/dense/MatMulMatMuldense_output/Reshape:output:00dense_output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
)dense_output/dense/BiasAdd/ReadVariableOpReadVariableOp2dense_output_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Æ
dense_output/dense/BiasAddBiasAdd#dense_output/dense/MatMul:product:01dense_output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’|
dense_output/dense/SigmoidSigmoid#dense_output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’q
dense_output/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’2      
dense_output/Reshape_1Reshapedense_output/dense/Sigmoid:y:0%dense_output/Reshape_1/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’2m
dense_output/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
dense_output/Reshape_2Reshapetf.concat_5/concat:output:0%dense_output/Reshape_2/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’r
IdentityIdentitydense_output/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2ź
NoOpNoOp*^dense_output/dense/BiasAdd/ReadVariableOp)^dense_output/dense/MatMul/ReadVariableOp%^dense_surface/BiasAdd/ReadVariableOp$^dense_surface/MatMul/ReadVariableOp%^gru/gru_cell_2/MatMul/ReadVariableOp'^gru/gru_cell_2/MatMul_1/ReadVariableOp^gru/gru_cell_2/ReadVariableOp
^gru/while'^gru_1/gru_cell_3/MatMul/ReadVariableOp)^gru_1/gru_cell_3/MatMul_1/ReadVariableOp ^gru_1/gru_cell_3/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 2V
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
:’’’’’’’’’1
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1
¾
Ŗ
while_cond_387067
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_387067___redundant_placeholder04
0while_while_cond_387067___redundant_placeholder14
0while_while_cond_387067___redundant_placeholder24
0while_while_cond_387067___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
6

A__inference_gru_1_layer_call_and_return_conditional_losses_387276

inputs$
gru_cell_3_387200:	%
gru_cell_3_387202:
%
gru_cell_3_387204:

identity¢"gru_cell_3/StatefulPartitionedCall¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ~
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:ź
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskÉ
"gru_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_3_387200gru_cell_3_387202gru_cell_3_387204*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_387199n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_3_387200gru_cell_3_387202gru_cell_3_387204*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_387212*
condR
while_cond_387211*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ģ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’s
NoOpNoOp#^gru_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’’’’’’’’’’: : : 2H
"gru_cell_3/StatefulPartitionedCall"gru_cell_3/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
E
·	
gru_1_while_body_388675(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0C
0gru_1_while_gru_cell_3_readvariableop_resource_0:	K
7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0:
M
9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0:

gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorA
.gru_1_while_gru_cell_3_readvariableop_resource:	I
5gru_1_while_gru_cell_3_matmul_readvariableop_resource:
K
7gru_1_while_gru_cell_3_matmul_1_readvariableop_resource:
¢,gru_1/while/gru_cell_3/MatMul/ReadVariableOp¢.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp¢%gru_1/while/gru_cell_3/ReadVariableOp
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Å
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
%gru_1/while/gru_cell_3/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru_1/while/gru_cell_3/unstackUnpack-gru_1/while/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num¦
,gru_1/while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Č
gru_1/while/gru_cell_3/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’®
gru_1/while/gru_cell_3/BiasAddBiasAdd'gru_1/while/gru_cell_3/MatMul:product:0'gru_1/while/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’q
&gru_1/while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’ė
gru_1/while/gru_cell_3/splitSplit/gru_1/while/gru_cell_3/split/split_dim:output:0'gru_1/while/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitŖ
.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Æ
gru_1/while/gru_cell_3/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’²
 gru_1/while/gru_cell_3/BiasAdd_1BiasAdd)gru_1/while/gru_cell_3/MatMul_1:product:0'gru_1/while/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’q
gru_1/while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’s
(gru_1/while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’„
gru_1/while/gru_cell_3/split_1SplitV)gru_1/while/gru_cell_3/BiasAdd_1:output:0%gru_1/while/gru_cell_3/Const:output:01gru_1/while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split¦
gru_1/while/gru_cell_3/addAddV2%gru_1/while/gru_cell_3/split:output:0'gru_1/while/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
gru_1/while/gru_cell_3/SigmoidSigmoidgru_1/while/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
gru_1/while/gru_cell_3/add_1AddV2%gru_1/while/gru_cell_3/split:output:1'gru_1/while/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
 gru_1/while/gru_cell_3/Sigmoid_1Sigmoid gru_1/while/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’£
gru_1/while/gru_cell_3/mulMul$gru_1/while/gru_cell_3/Sigmoid_1:y:0'gru_1/while/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/while/gru_cell_3/add_2AddV2%gru_1/while/gru_cell_3/split:output:2gru_1/while/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_1/while/gru_cell_3/ReluRelu gru_1/while/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/while/gru_cell_3/mul_1Mul"gru_1/while/gru_cell_3/Sigmoid:y:0gru_1_while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’a
gru_1/while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/while/gru_cell_3/subSub%gru_1/while/gru_cell_3/sub/x:output:0"gru_1/while/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’”
gru_1/while/gru_cell_3/mul_2Mulgru_1/while/gru_cell_3/sub:z:0)gru_1/while/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/while/gru_cell_3/add_3AddV2 gru_1/while/gru_cell_3/mul_1:z:0 gru_1/while/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ū
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅS
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
: 
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_4Identity gru_1/while/gru_cell_3/add_3:z:0^gru_1/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ś
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
gru_1_while_identity_4gru_1/while/Identity_4:output:0"Ą
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2\
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 


ó
A__inference_dense_layer_call_and_return_conditional_losses_387350

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¬
Ü
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_390699

inputs
states_0*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’É
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:’’’’’’’’’J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’:’’’’’’’’’: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0

¹
&__inference_gru_1_layer_call_fn_389833

inputs
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCallė
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:’’’’’’’’’2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_387754t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:’’’’’’’’’2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ą#
Ų
A__inference_model_layer_call_and_return_conditional_losses_388231

inputs
inputs_1

gru_388195:	

gru_388197:	

gru_388199:
(
dense_surface_388205:
#
dense_surface_388207:	
gru_1_388214:	 
gru_1_388216:
 
gru_1_388218:
&
dense_output_388223:	!
dense_output_388225:
identity¢$dense_output/StatefulPartitionedCall¢%dense_surface/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCall
gru/StatefulPartitionedCallStatefulPartitionedCallinputs
gru_388195
gru_388197
gru_388199*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:’’’’’’’’’1:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_387929Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¬
tf.concat_3/concatConcatV2$gru/StatefulPartitionedCall:output:1inputs_1 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
%dense_surface/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_surface_388205dense_surface_388207*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_387590o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’      «
tf.reshape_1/ReshapeReshape.dense_surface/StatefulPartitionedCall:output:0#tf.reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
tf.concat_4/concatConcatV2$gru/StatefulPartitionedCall:output:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2
gru_1/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0gru_1_388214gru_1_388216gru_1_388218*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:’’’’’’’’’2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_388103Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0&gru_1/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2
$dense_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_output_388223dense_output_388225*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_387381k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’
IdentityIdentity-dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2Ó
NoOpNoOp%^dense_output/StatefulPartitionedCall&^dense_surface/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall2N
%dense_surface/StatefulPartitionedCall%dense_surface/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’1
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
øB
ł
gru_while_body_388511$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0A
.gru_while_gru_cell_2_readvariableop_resource_0:	H
5gru_while_gru_cell_2_matmul_readvariableop_resource_0:	K
7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:

gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor?
,gru_while_gru_cell_2_readvariableop_resource:	F
3gru_while_gru_cell_2_matmul_readvariableop_resource:	I
5gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢*gru/while/gru_cell_2/MatMul/ReadVariableOp¢,gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢#gru/while/gru_cell_2/ReadVariableOp
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ŗ
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
#gru/while/gru_cell_2/ReadVariableOpReadVariableOp.gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru/while/gru_cell_2/unstackUnpack+gru/while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num”
*gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp5gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ā
gru/while/gru_cell_2/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:02gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
gru/while/gru_cell_2/BiasAddBiasAdd%gru/while/gru_cell_2/MatMul:product:0%gru/while/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’o
$gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’å
gru/while/gru_cell_2/splitSplit-gru/while/gru_cell_2/split/split_dim:output:0%gru/while/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split¦
,gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp7gru_while_gru_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0©
gru/while/gru_cell_2/MatMul_1MatMulgru_while_placeholder_24gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¬
gru/while/gru_cell_2/BiasAdd_1BiasAdd'gru/while/gru_cell_2/MatMul_1:product:0%gru/while/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’o
gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’q
&gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
gru/while/gru_cell_2/split_1SplitV'gru/while/gru_cell_2/BiasAdd_1:output:0#gru/while/gru_cell_2/Const:output:0/gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split 
gru/while/gru_cell_2/addAddV2#gru/while/gru_cell_2/split:output:0%gru/while/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru/while/gru_cell_2/SigmoidSigmoidgru/while/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’¢
gru/while/gru_cell_2/add_1AddV2#gru/while/gru_cell_2/split:output:1%gru/while/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’|
gru/while/gru_cell_2/Sigmoid_1Sigmoidgru/while/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/mulMul"gru/while/gru_cell_2/Sigmoid_1:y:0%gru/while/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/add_2AddV2#gru/while/gru_cell_2/split:output:2gru/while/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
gru/while/gru_cell_2/ReluRelugru/while/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/mul_1Mul gru/while/gru_cell_2/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’_
gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/while/gru_cell_2/subSub#gru/while/gru_cell_2/sub/x:output:0 gru/while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/mul_2Mulgru/while/gru_cell_2/sub:z:0'gru/while/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/while/gru_cell_2/add_3AddV2gru/while/gru_cell_2/mul_1:z:0gru/while/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ó
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅQ
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
: 
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: 
gru/while/Identity_4Identitygru/while/gru_cell_2/add_3:z:0^gru/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ņ
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
gru_while_identity_4gru/while/Identity_4:output:0"ø
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2X
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
ē	
Č
$__inference_gru_layer_call_fn_389125
inputs_0
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:’’’’’’’’’’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_386787}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
ģ	

gru_1_while_cond_389007(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_389007___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_389007___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_389007___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_389007___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::P L
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
ŁN

A__inference_gru_1_layer_call_and_return_conditional_losses_387754

inputs5
"gru_cell_3_readvariableop_resource:	=
)gru_cell_3_matmul_readvariableop_resource:
?
+gru_cell_3_matmul_1_readvariableop_resource:

identity¢ gru_cell_3/MatMul/ReadVariableOp¢"gru_cell_3/MatMul_1/ReadVariableOp¢gru_cell_3/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: u
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*,
_output_shapes
:2’’’’’’’’’
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:ź
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_387665*
condR
while_cond_387664*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ć
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’2²
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:’’’’’’’’’2: : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
÷#
ß
A__inference_model_layer_call_and_return_conditional_losses_387772
inputs_main

inputs_aux

gru_387569:	

gru_387571:	

gru_387573:
(
dense_surface_387591:
#
dense_surface_387593:	
gru_1_387755:	 
gru_1_387757:
 
gru_1_387759:
&
dense_output_387764:	!
dense_output_387766:
identity¢$dense_output/StatefulPartitionedCall¢%dense_surface/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCall
gru/StatefulPartitionedCallStatefulPartitionedCallinputs_main
gru_387569
gru_387571
gru_387573*
Tin
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:’’’’’’’’’1:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_387568Y
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :®
tf.concat_3/concatConcatV2$gru/StatefulPartitionedCall:output:1
inputs_aux tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
%dense_surface/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_surface_387591dense_surface_387593*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_surface_layer_call_and_return_conditional_losses_387590o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’      «
tf.reshape_1/ReshapeReshape.dense_surface/StatefulPartitionedCall:output:0#tf.reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
tf.concat_4/concatConcatV2$gru/StatefulPartitionedCall:output:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2
gru_1/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0gru_1_387755gru_1_387757gru_1_387759*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:’’’’’’’’’2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_387754Y
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Å
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0&gru_1/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2
$dense_output/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_output_387764dense_output_387766*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_387361k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’
IdentityIdentity-dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2Ó
NoOpNoOp%^dense_output/StatefulPartitionedCall&^dense_surface/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall2N
%dense_surface/StatefulPartitionedCall%dense_surface/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:X T
+
_output_shapes
:’’’’’’’’’1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
inputs_aux
Ļ 
°
while_body_386865
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_2_386887_0:	,
while_gru_cell_2_386889_0:	-
while_gru_cell_2_386891_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_2_386887:	*
while_gru_cell_2_386889:	+
while_gru_cell_2_386891:
¢(while/gru_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_386887_0while_gru_cell_2_386889_0while_gru_cell_2_386891_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_386852Ś
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w

while/NoOpNoOp)^while/gru_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_2_386887while_gru_cell_2_386887_0"4
while_gru_cell_2_386889while_gru_cell_2_386889_0"4
while_gru_cell_2_386891while_gru_cell_2_386891_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2T
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
¾
Ŗ
while_cond_389689
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_389689___redundant_placeholder04
0while_while_cond_389689___redundant_placeholder14
0while_while_cond_389689___redundant_placeholder24
0while_while_cond_389689___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
¼

Ū
+__inference_gru_cell_2_layer_call_fn_390554

inputs
states_0
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_386852p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0
½

&__inference_model_layer_call_fn_388420
inputs_0
inputs_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:

	unknown_7:	
	unknown_8:
identity¢StatefulPartitionedCallŅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_388165s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:’’’’’’’’’1
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1
Æ=

while_body_389536
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_2_readvariableop_resource_0:	D
1while_gru_cell_2_matmul_readvariableop_resource_0:	G
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_2_readvariableop_resource:	B
/while_gru_cell_2_matmul_readvariableop_resource:	E
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¶
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
³=

while_body_387665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_3_readvariableop_resource_0:	E
1while_gru_cell_3_matmul_readvariableop_resource_0:
G
3while_gru_cell_3_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_3_readvariableop_resource:	C
/while_gru_cell_3_matmul_readvariableop_resource:
E
1while_gru_cell_3_matmul_1_readvariableop_resource:
¢&while/gru_cell_3/MatMul/ReadVariableOp¢(while/gru_cell_3/MatMul_1/ReadVariableOp¢while/gru_cell_3/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
ģ	

gru_1_while_cond_388674(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_388674___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_388674___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_388674___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_388674___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::P L
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
ÓL
ė

model_gru_1_while_body_3865384
0model_gru_1_while_model_gru_1_while_loop_counter:
6model_gru_1_while_model_gru_1_while_maximum_iterations!
model_gru_1_while_placeholder#
model_gru_1_while_placeholder_1#
model_gru_1_while_placeholder_23
/model_gru_1_while_model_gru_1_strided_slice_1_0o
kmodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensor_0I
6model_gru_1_while_gru_cell_3_readvariableop_resource_0:	Q
=model_gru_1_while_gru_cell_3_matmul_readvariableop_resource_0:
S
?model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0:

model_gru_1_while_identity 
model_gru_1_while_identity_1 
model_gru_1_while_identity_2 
model_gru_1_while_identity_3 
model_gru_1_while_identity_41
-model_gru_1_while_model_gru_1_strided_slice_1m
imodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensorG
4model_gru_1_while_gru_cell_3_readvariableop_resource:	O
;model_gru_1_while_gru_cell_3_matmul_readvariableop_resource:
Q
=model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource:
¢2model/gru_1/while/gru_cell_3/MatMul/ReadVariableOp¢4model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp¢+model/gru_1/while/gru_cell_3/ReadVariableOp
Cmodel/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ć
5model/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkmodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensor_0model_gru_1_while_placeholderLmodel/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0£
+model/gru_1/while/gru_cell_3/ReadVariableOpReadVariableOp6model_gru_1_while_gru_cell_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
$model/gru_1/while/gru_cell_3/unstackUnpack3model/gru_1/while/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num²
2model/gru_1/while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp=model_gru_1_while_gru_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ś
#model/gru_1/while/gru_cell_3/MatMulMatMul<model/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0:model/gru_1/while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Ą
$model/gru_1/while/gru_cell_3/BiasAddBiasAdd-model/gru_1/while/gru_cell_3/MatMul:product:0-model/gru_1/while/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’w
,model/gru_1/while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’ż
"model/gru_1/while/gru_cell_3/splitSplit5model/gru_1/while/gru_cell_3/split/split_dim:output:0-model/gru_1/while/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split¶
4model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp?model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Į
%model/gru_1/while/gru_cell_3/MatMul_1MatMulmodel_gru_1_while_placeholder_2<model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Ä
&model/gru_1/while/gru_cell_3/BiasAdd_1BiasAdd/model/gru_1/while/gru_cell_3/MatMul_1:product:0-model/gru_1/while/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’w
"model/gru_1/while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’y
.model/gru_1/while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’½
$model/gru_1/while/gru_cell_3/split_1SplitV/model/gru_1/while/gru_cell_3/BiasAdd_1:output:0+model/gru_1/while/gru_cell_3/Const:output:07model/gru_1/while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitø
 model/gru_1/while/gru_cell_3/addAddV2+model/gru_1/while/gru_cell_3/split:output:0-model/gru_1/while/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
$model/gru_1/while/gru_cell_3/SigmoidSigmoid$model/gru_1/while/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’ŗ
"model/gru_1/while/gru_cell_3/add_1AddV2+model/gru_1/while/gru_cell_3/split:output:1-model/gru_1/while/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
&model/gru_1/while/gru_cell_3/Sigmoid_1Sigmoid&model/gru_1/while/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’µ
 model/gru_1/while/gru_cell_3/mulMul*model/gru_1/while/gru_cell_3/Sigmoid_1:y:0-model/gru_1/while/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’±
"model/gru_1/while/gru_cell_3/add_2AddV2+model/gru_1/while/gru_cell_3/split:output:2$model/gru_1/while/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’
!model/gru_1/while/gru_cell_3/ReluRelu&model/gru_1/while/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’§
"model/gru_1/while/gru_cell_3/mul_1Mul(model/gru_1/while/gru_cell_3/Sigmoid:y:0model_gru_1_while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’g
"model/gru_1/while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
 model/gru_1/while/gru_cell_3/subSub+model/gru_1/while/gru_cell_3/sub/x:output:0(model/gru_1/while/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’³
"model/gru_1/while/gru_cell_3/mul_2Mul$model/gru_1/while/gru_cell_3/sub:z:0/model/gru_1/while/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’®
"model/gru_1/while/gru_cell_3/add_3AddV2&model/gru_1/while/gru_cell_3/mul_1:z:0&model/gru_1/while/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’ó
6model/gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmodel_gru_1_while_placeholder_1model_gru_1_while_placeholder&model/gru_1/while/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅY
model/gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model/gru_1/while/addAddV2model_gru_1_while_placeholder model/gru_1/while/add/y:output:0*
T0*
_output_shapes
: [
model/gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
model/gru_1/while/add_1AddV20model_gru_1_while_model_gru_1_while_loop_counter"model/gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: }
model/gru_1/while/IdentityIdentitymodel/gru_1/while/add_1:z:0^model/gru_1/while/NoOp*
T0*
_output_shapes
: 
model/gru_1/while/Identity_1Identity6model_gru_1_while_model_gru_1_while_maximum_iterations^model/gru_1/while/NoOp*
T0*
_output_shapes
: }
model/gru_1/while/Identity_2Identitymodel/gru_1/while/add:z:0^model/gru_1/while/NoOp*
T0*
_output_shapes
: Ŗ
model/gru_1/while/Identity_3IdentityFmodel/gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/gru_1/while/NoOp*
T0*
_output_shapes
: 
model/gru_1/while/Identity_4Identity&model/gru_1/while/gru_cell_3/add_3:z:0^model/gru_1/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’ņ
model/gru_1/while/NoOpNoOp3^model/gru_1/while/gru_cell_3/MatMul/ReadVariableOp5^model/gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp,^model/gru_1/while/gru_cell_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
=model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource?model_gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0"|
;model_gru_1_while_gru_cell_3_matmul_readvariableop_resource=model_gru_1_while_gru_cell_3_matmul_readvariableop_resource_0"n
4model_gru_1_while_gru_cell_3_readvariableop_resource6model_gru_1_while_gru_cell_3_readvariableop_resource_0"A
model_gru_1_while_identity#model/gru_1/while/Identity:output:0"E
model_gru_1_while_identity_1%model/gru_1/while/Identity_1:output:0"E
model_gru_1_while_identity_2%model/gru_1/while/Identity_2:output:0"E
model_gru_1_while_identity_3%model/gru_1/while/Identity_3:output:0"E
model_gru_1_while_identity_4%model/gru_1/while/Identity_4:output:0"`
-model_gru_1_while_model_gru_1_strided_slice_1/model_gru_1_while_model_gru_1_strided_slice_1_0"Ų
imodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensorkmodel_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2h
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
Æ=

while_body_389228
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_2_readvariableop_resource_0:	D
1while_gru_cell_2_matmul_readvariableop_resource_0:	G
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_2_readvariableop_resource:	B
/while_gru_cell_2_matmul_readvariableop_resource:	E
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¶
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
¬

ż
I__inference_dense_surface_layer_call_and_return_conditional_losses_389800

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


-__inference_dense_output_layer_call_fn_390482

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallķ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dense_output_layer_call_and_return_conditional_losses_387381|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
æ

Ü
+__inference_gru_cell_3_layer_call_fn_390646

inputs
states_0
unknown:	
	unknown_0:

	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_387055p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’:’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0

Ä
H__inference_dense_output_layer_call_and_return_conditional_losses_387361

inputs
dense_387351:	
dense_387353:
identity¢dense/StatefulPartitionedCallI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
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
valueB"’’’’   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’ń
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_387351dense_387353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_387350\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’’’’’’’’’’: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
³=

while_body_390375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_3_readvariableop_resource_0:	E
1while_gru_cell_3_matmul_readvariableop_resource_0:
G
3while_gru_cell_3_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_3_readvariableop_resource:	C
/while_gru_cell_3_matmul_readvariableop_resource:
E
1while_gru_cell_3_matmul_1_readvariableop_resource:
¢&while/gru_cell_3/MatMul/ReadVariableOp¢(while/gru_cell_3/MatMul_1/ReadVariableOp¢while/gru_cell_3/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
¤
Ś
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_387199

inputs

states*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’É
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:’’’’’’’’’J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’:’’’’’’’’’: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:PL
(
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates
N

?__inference_gru_layer_call_and_return_conditional_losses_387568

inputs5
"gru_cell_2_readvariableop_resource:	<
)gru_cell_2_matmul_readvariableop_resource:	?
+gru_cell_2_matmul_1_readvariableop_resource:

identity

identity_1¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:1’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_387478*
condR
while_cond_387477*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ć
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:1’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’1[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’1`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:’’’’’’’’’²
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’1: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’1
 
_user_specified_nameinputs
¾
Ŗ
while_cond_387211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_387211___redundant_placeholder04
0while_while_cond_387211___redundant_placeholder14
0while_while_cond_387211___redundant_placeholder24
0while_while_cond_387211___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
¼

Ū
+__inference_gru_cell_2_layer_call_fn_390540

inputs
states_0
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_386709p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0
¾
Ŗ
while_cond_390219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390219___redundant_placeholder04
0while_while_cond_390219___redundant_placeholder14
0while_while_cond_390219___redundant_placeholder24
0while_while_cond_390219___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
 
Ł
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_386709

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’É
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:’’’’’’’’’J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:PL
(
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates
Æ=

while_body_389690
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_2_readvariableop_resource_0:	D
1while_gru_cell_2_matmul_readvariableop_resource_0:	G
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_2_readvariableop_resource:	B
/while_gru_cell_2_matmul_readvariableop_resource:	E
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¶
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
 
Ł
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_386852

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’É
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:’’’’’’’’’J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:PL
(
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates
¾
Ŗ
while_cond_386864
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_386864___redundant_placeholder04
0while_while_cond_386864___redundant_placeholder14
0while_while_cond_386864___redundant_placeholder24
0while_while_cond_386864___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
ĪN

?__inference_gru_layer_call_and_return_conditional_losses_389472
inputs_05
"gru_cell_2_readvariableop_resource:	<
)gru_cell_2_matmul_readvariableop_resource:	?
+gru_cell_2_matmul_1_readvariableop_resource:

identity

identity_1¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_389382*
condR
while_cond_389381*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ģ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:’’’’’’’’’²
NoOpNoOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
¾
Ŗ
while_cond_389535
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_389535___redundant_placeholder04
0while_while_cond_389535___redundant_placeholder14
0while_while_cond_389535___redundant_placeholder24
0while_while_cond_389535___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
E
·	
gru_1_while_body_389008(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0C
0gru_1_while_gru_cell_3_readvariableop_resource_0:	K
7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0:
M
9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0:

gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorA
.gru_1_while_gru_cell_3_readvariableop_resource:	I
5gru_1_while_gru_cell_3_matmul_readvariableop_resource:
K
7gru_1_while_gru_cell_3_matmul_1_readvariableop_resource:
¢,gru_1/while/gru_cell_3/MatMul/ReadVariableOp¢.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp¢%gru_1/while/gru_cell_3/ReadVariableOp
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Å
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
%gru_1/while/gru_cell_3/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru_1/while/gru_cell_3/unstackUnpack-gru_1/while/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num¦
,gru_1/while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Č
gru_1/while/gru_cell_3/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’®
gru_1/while/gru_cell_3/BiasAddBiasAdd'gru_1/while/gru_cell_3/MatMul:product:0'gru_1/while/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’q
&gru_1/while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’ė
gru_1/while/gru_cell_3/splitSplit/gru_1/while/gru_cell_3/split/split_dim:output:0'gru_1/while/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitŖ
.gru_1/while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Æ
gru_1/while/gru_cell_3/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’²
 gru_1/while/gru_cell_3/BiasAdd_1BiasAdd)gru_1/while/gru_cell_3/MatMul_1:product:0'gru_1/while/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’q
gru_1/while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’s
(gru_1/while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’„
gru_1/while/gru_cell_3/split_1SplitV)gru_1/while/gru_cell_3/BiasAdd_1:output:0%gru_1/while/gru_cell_3/Const:output:01gru_1/while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split¦
gru_1/while/gru_cell_3/addAddV2%gru_1/while/gru_cell_3/split:output:0'gru_1/while/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
gru_1/while/gru_cell_3/SigmoidSigmoidgru_1/while/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
gru_1/while/gru_cell_3/add_1AddV2%gru_1/while/gru_cell_3/split:output:1'gru_1/while/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
 gru_1/while/gru_cell_3/Sigmoid_1Sigmoid gru_1/while/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’£
gru_1/while/gru_cell_3/mulMul$gru_1/while/gru_cell_3/Sigmoid_1:y:0'gru_1/while/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/while/gru_cell_3/add_2AddV2%gru_1/while/gru_cell_3/split:output:2gru_1/while/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_1/while/gru_cell_3/ReluRelu gru_1/while/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/while/gru_cell_3/mul_1Mul"gru_1/while/gru_cell_3/Sigmoid:y:0gru_1_while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’a
gru_1/while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/while/gru_cell_3/subSub%gru_1/while/gru_cell_3/sub/x:output:0"gru_1/while/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’”
gru_1/while/gru_cell_3/mul_2Mulgru_1/while/gru_cell_3/sub:z:0)gru_1/while/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/while/gru_cell_3/add_3AddV2 gru_1/while/gru_cell_3/mul_1:z:0 gru_1/while/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ū
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅS
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
: 
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_4Identity gru_1/while/gru_cell_3/add_3:z:0^gru_1/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ś
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
gru_1_while_identity_4gru_1/while/Identity_4:output:0"Ą
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2\
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
¬

ż
I__inference_dense_surface_layer_call_and_return_conditional_losses_387590

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¾
Ŗ
while_cond_387477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_387477___redundant_placeholder04
0while_while_cond_387477___redundant_placeholder14
0while_while_cond_387477___redundant_placeholder24
0while_while_cond_387477___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
²
»
&__inference_gru_1_layer_call_fn_389822
inputs_0
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_387276}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
Ø
Ū
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_390632

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’É
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:’’’’’’’’’J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0
³=

while_body_388014
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_3_readvariableop_resource_0:	E
1while_gru_cell_3_matmul_readvariableop_resource_0:
G
3while_gru_cell_3_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_3_readvariableop_resource:	C
/while_gru_cell_3_matmul_readvariableop_resource:
E
1while_gru_cell_3_matmul_1_readvariableop_resource:
¢&while/gru_cell_3/MatMul/ReadVariableOp¢(while/gru_cell_3/MatMul_1/ReadVariableOp¢while/gru_cell_3/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
Æ=

while_body_387478
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_2_readvariableop_resource_0:	D
1while_gru_cell_2_matmul_readvariableop_resource_0:	G
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_2_readvariableop_resource:	B
/while_gru_cell_2_matmul_readvariableop_resource:	E
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¶
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 

¹
&__inference_gru_1_layer_call_fn_389844

inputs
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCallė
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:’’’’’’’’’2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_388103t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:’’’’’’’’’2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:’’’’’’’’’2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
¾
Ŗ
while_cond_389227
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_389227___redundant_placeholder04
0while_while_cond_389227___redundant_placeholder14
0while_while_cond_389227___redundant_placeholder24
0while_while_cond_389227___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
Æ=

while_body_389382
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_2_readvariableop_resource_0:	D
1while_gru_cell_2_matmul_readvariableop_resource_0:	G
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_2_readvariableop_resource:	B
/while_gru_cell_2_matmul_readvariableop_resource:	E
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¶
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
³=

while_body_389910
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_3_readvariableop_resource_0:	E
1while_gru_cell_3_matmul_readvariableop_resource_0:
G
3while_gru_cell_3_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_3_readvariableop_resource:	C
/while_gru_cell_3_matmul_readvariableop_resource:
E
1while_gru_cell_3_matmul_1_readvariableop_resource:
¢&while/gru_cell_3/MatMul/ReadVariableOp¢(while/gru_cell_3/MatMul_1/ReadVariableOp¢while/gru_cell_3/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’*
element_dtype0
while/gru_cell_3/ReadVariableOpReadVariableOp*while_gru_cell_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_3/unstackUnpack'while/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_3/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/BiasAddBiasAdd!while/gru_cell_3/MatMul:product:0!while/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 while/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
while/gru_cell_3/splitSplit)while/gru_cell_3/split/split_dim:output:0!while/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(while/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_3_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_3/MatMul_1MatMulwhile_placeholder_20while/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
while/gru_cell_3/BiasAdd_1BiasAdd#while/gru_cell_3/MatMul_1:product:0!while/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
while/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"while/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
while/gru_cell_3/split_1SplitV#while/gru_cell_3/BiasAdd_1:output:0while/gru_cell_3/Const:output:0+while/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
while/gru_cell_3/addAddV2while/gru_cell_3/split:output:0!while/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
while/gru_cell_3/SigmoidSigmoidwhile/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_1AddV2while/gru_cell_3/split:output:1!while/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
while/gru_cell_3/Sigmoid_1Sigmoidwhile/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mulMulwhile/gru_cell_3/Sigmoid_1:y:0!while/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_2AddV2while/gru_cell_3/split:output:2while/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/gru_cell_3/ReluReluwhile/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_1Mulwhile/gru_cell_3/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’[
while/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_3/subSubwhile/gru_cell_3/sub/x:output:0while/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/mul_2Mulwhile/gru_cell_3/sub:z:0#while/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
while/gru_cell_3/add_3AddV2while/gru_cell_3/mul_1:z:0while/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_3/add_3:z:0*
_output_shapes
: *
element_dtype0:éčŅM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ā

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
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :’’’’’’’’’: : : : : 2P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: 
¾
Ŗ
while_cond_389381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_389381___redundant_placeholder04
0while_while_cond_389381___redundant_placeholder14
0while_while_cond_389381___redundant_placeholder24
0while_while_cond_389381___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
Ŗ

$__inference_signature_wrapper_388394

inputs_aux
inputs_main
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:

	unknown_7:	
	unknown_8:
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputs_main
inputs_auxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_386642s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’:’’’’’’’’’1: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
inputs_aux:XT
+
_output_shapes
:’’’’’’’’’1
%
_user_specified_nameinputs_main
¾
Ŗ
while_cond_387664
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_387664___redundant_placeholder04
0while_while_cond_387664___redundant_placeholder14
0while_while_cond_387664___redundant_placeholder24
0while_while_cond_387664___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
Ū


!__inference__wrapped_model_386642
inputs_main

inputs_aux?
,model_gru_gru_cell_2_readvariableop_resource:	F
3model_gru_gru_cell_2_matmul_readvariableop_resource:	I
5model_gru_gru_cell_2_matmul_1_readvariableop_resource:
F
2model_dense_surface_matmul_readvariableop_resource:
B
3model_dense_surface_biasadd_readvariableop_resource:	A
.model_gru_1_gru_cell_3_readvariableop_resource:	I
5model_gru_1_gru_cell_3_matmul_readvariableop_resource:
K
7model_gru_1_gru_cell_3_matmul_1_readvariableop_resource:
J
7model_dense_output_dense_matmul_readvariableop_resource:	F
8model_dense_output_dense_biasadd_readvariableop_resource:
identity¢/model/dense_output/dense/BiasAdd/ReadVariableOp¢.model/dense_output/dense/MatMul/ReadVariableOp¢*model/dense_surface/BiasAdd/ReadVariableOp¢)model/dense_surface/MatMul/ReadVariableOp¢*model/gru/gru_cell_2/MatMul/ReadVariableOp¢,model/gru/gru_cell_2/MatMul_1/ReadVariableOp¢#model/gru/gru_cell_2/ReadVariableOp¢model/gru/while¢,model/gru_1/gru_cell_3/MatMul/ReadVariableOp¢.model/gru_1/gru_cell_3/MatMul_1/ReadVariableOp¢%model/gru_1/gru_cell_3/ReadVariableOp¢model/gru_1/whileX
model/gru/ShapeShapeinputs_main*
T0*
_output_shapes
::ķĻg
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
valueB:
model/gru/strided_sliceStridedSlicemodel/gru/Shape:output:0&model/gru/strided_slice/stack:output:0(model/gru/strided_slice/stack_1:output:0(model/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
model/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
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
 *    
model/gru/zerosFillmodel/gru/zeros/packed:output:0model/gru/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’m
model/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
model/gru/transpose	Transposeinputs_main!model/gru/transpose/perm:output:0*
T0*+
_output_shapes
:1’’’’’’’’’f
model/gru/Shape_1Shapemodel/gru/transpose:y:0*
T0*
_output_shapes
::ķĻi
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
valueB:
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
’’’’’’’’’Ņ
model/gru/TensorArrayV2TensorListReserve.model/gru/TensorArrayV2/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
?model/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ž
1model/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/gru/transpose:y:0Hmodel/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅi
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
valueB:
model/gru/strided_slice_2StridedSlicemodel/gru/transpose:y:0(model/gru/strided_slice_2/stack:output:0*model/gru/strided_slice_2/stack_1:output:0*model/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask
#model/gru/gru_cell_2/ReadVariableOpReadVariableOp,model_gru_gru_cell_2_readvariableop_resource*
_output_shapes
:	*
dtype0
model/gru/gru_cell_2/unstackUnpack+model/gru/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
*model/gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp3model_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0°
model/gru/gru_cell_2/MatMulMatMul"model/gru/strided_slice_2:output:02model/gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
model/gru/gru_cell_2/BiasAddBiasAdd%model/gru/gru_cell_2/MatMul:product:0%model/gru/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’o
$model/gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’å
model/gru/gru_cell_2/splitSplit-model/gru/gru_cell_2/split/split_dim:output:0%model/gru/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split¤
,model/gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp5model_gru_gru_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Ŗ
model/gru/gru_cell_2/MatMul_1MatMulmodel/gru/zeros:output:04model/gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¬
model/gru/gru_cell_2/BiasAdd_1BiasAdd'model/gru/gru_cell_2/MatMul_1:product:0%model/gru/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’o
model/gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’q
&model/gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
model/gru/gru_cell_2/split_1SplitV'model/gru/gru_cell_2/BiasAdd_1:output:0#model/gru/gru_cell_2/Const:output:0/model/gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split 
model/gru/gru_cell_2/addAddV2#model/gru/gru_cell_2/split:output:0%model/gru/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’x
model/gru/gru_cell_2/SigmoidSigmoidmodel/gru/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’¢
model/gru/gru_cell_2/add_1AddV2#model/gru/gru_cell_2/split:output:1%model/gru/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’|
model/gru/gru_cell_2/Sigmoid_1Sigmoidmodel/gru/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/gru/gru_cell_2/mulMul"model/gru/gru_cell_2/Sigmoid_1:y:0%model/gru/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
model/gru/gru_cell_2/add_2AddV2#model/gru/gru_cell_2/split:output:2model/gru/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
model/gru/gru_cell_2/ReluRelumodel/gru/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/gru/gru_cell_2/mul_1Mul model/gru/gru_cell_2/Sigmoid:y:0model/gru/zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’_
model/gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/gru/gru_cell_2/subSub#model/gru/gru_cell_2/sub/x:output:0 model/gru/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
model/gru/gru_cell_2/mul_2Mulmodel/gru/gru_cell_2/sub:z:0'model/gru/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
model/gru/gru_cell_2/add_3AddV2model/gru/gru_cell_2/mul_1:z:0model/gru/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’x
'model/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ö
model/gru/TensorArrayV2_1TensorListReserve0model/gru/TensorArrayV2_1/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅP
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
’’’’’’’’’^
model/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : æ
model/gru/whileWhile%model/gru/while/loop_counter:output:0+model/gru/while/maximum_iterations:output:0model/gru/time:output:0"model/gru/TensorArrayV2_1:handle:0model/gru/zeros:output:0"model/gru/strided_slice_1:output:0Amodel/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0,model_gru_gru_cell_2_readvariableop_resource3model_gru_gru_cell_2_matmul_readvariableop_resource5model_gru_gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *'
bodyR
model_gru_while_body_386374*'
condR
model_gru_while_cond_386373*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
:model/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   į
,model/gru/TensorArrayV2Stack/TensorListStackTensorListStackmodel/gru/while:output:3Cmodel/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:1’’’’’’’’’*
element_dtype0r
model/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’k
!model/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!model/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŗ
model/gru/strided_slice_3StridedSlice5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0(model/gru/strided_slice_3/stack:output:0*model/gru/strided_slice_3/stack_1:output:0*model/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_masko
model/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
model/gru/transpose_1	Transpose5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0#model/gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’1e
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
value	B :®
model/tf.concat_3/concatConcatV2model/gru/while:output:4
inputs_aux&model/tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
)model/dense_surface/MatMul/ReadVariableOpReadVariableOp2model_dense_surface_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0­
model/dense_surface/MatMulMatMul!model/tf.concat_3/concat:output:01model/dense_surface/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
*model/dense_surface/BiasAdd/ReadVariableOpReadVariableOp3model_dense_surface_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0³
model/dense_surface/BiasAddBiasAdd$model/dense_surface/MatMul:product:02model/dense_surface/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’y
model/dense_surface/ReluRelu$model/dense_surface/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
 model/tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’      Æ
model/tf.reshape_1/ReshapeReshape&model/dense_surface/Relu:activations:0)model/tf.reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’_
model/tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ģ
model/tf.concat_4/concatConcatV2model/gru/transpose_1:y:0#model/tf.reshape_1/Reshape:output:0&model/tf.concat_4/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2p
model/gru_1/ShapeShape!model/tf.concat_4/concat:output:0*
T0*
_output_shapes
::ķĻi
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
valueB:
model/gru_1/strided_sliceStridedSlicemodel/gru_1/Shape:output:0(model/gru_1/strided_slice/stack:output:0*model/gru_1/strided_slice/stack_1:output:0*model/gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
model/gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
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
 *    
model/gru_1/zerosFill!model/gru_1/zeros/packed:output:0 model/gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’o
model/gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ”
model/gru_1/transpose	Transpose!model/tf.concat_4/concat:output:0#model/gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:2’’’’’’’’’j
model/gru_1/Shape_1Shapemodel/gru_1/transpose:y:0*
T0*
_output_shapes
::ķĻk
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
valueB:
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
’’’’’’’’’Ų
model/gru_1/TensorArrayV2TensorListReserve0model/gru_1/TensorArrayV2/element_shape:output:0$model/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅd
model/gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
model/gru_1/ReverseV2	ReverseV2model/gru_1/transpose:y:0#model/gru_1/ReverseV2/axis:output:0*
T0*,
_output_shapes
:2’’’’’’’’’
Amodel/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
3model/gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/gru_1/ReverseV2:output:0Jmodel/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅk
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
valueB:¦
model/gru_1/strided_slice_2StridedSlicemodel/gru_1/transpose:y:0*model/gru_1/strided_slice_2/stack:output:0,model/gru_1/strided_slice_2/stack_1:output:0,model/gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask
%model/gru_1/gru_cell_3/ReadVariableOpReadVariableOp.model_gru_1_gru_cell_3_readvariableop_resource*
_output_shapes
:	*
dtype0
model/gru_1/gru_cell_3/unstackUnpack-model/gru_1/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num¤
,model/gru_1/gru_cell_3/MatMul/ReadVariableOpReadVariableOp5model_gru_1_gru_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¶
model/gru_1/gru_cell_3/MatMulMatMul$model/gru_1/strided_slice_2:output:04model/gru_1/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’®
model/gru_1/gru_cell_3/BiasAddBiasAdd'model/gru_1/gru_cell_3/MatMul:product:0'model/gru_1/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’q
&model/gru_1/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’ė
model/gru_1/gru_cell_3/splitSplit/model/gru_1/gru_cell_3/split/split_dim:output:0'model/gru_1/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitØ
.model/gru_1/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp7model_gru_1_gru_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0°
model/gru_1/gru_cell_3/MatMul_1MatMulmodel/gru_1/zeros:output:06model/gru_1/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’²
 model/gru_1/gru_cell_3/BiasAdd_1BiasAdd)model/gru_1/gru_cell_3/MatMul_1:product:0'model/gru_1/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’q
model/gru_1/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’s
(model/gru_1/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’„
model/gru_1/gru_cell_3/split_1SplitV)model/gru_1/gru_cell_3/BiasAdd_1:output:0%model/gru_1/gru_cell_3/Const:output:01model/gru_1/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split¦
model/gru_1/gru_cell_3/addAddV2%model/gru_1/gru_cell_3/split:output:0'model/gru_1/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
model/gru_1/gru_cell_3/SigmoidSigmoidmodel/gru_1/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
model/gru_1/gru_cell_3/add_1AddV2%model/gru_1/gru_cell_3/split:output:1'model/gru_1/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
 model/gru_1/gru_cell_3/Sigmoid_1Sigmoid model/gru_1/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’£
model/gru_1/gru_cell_3/mulMul$model/gru_1/gru_cell_3/Sigmoid_1:y:0'model/gru_1/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
model/gru_1/gru_cell_3/add_2AddV2%model/gru_1/gru_cell_3/split:output:2model/gru_1/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’x
model/gru_1/gru_cell_3/ReluRelu model/gru_1/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/gru_1/gru_cell_3/mul_1Mul"model/gru_1/gru_cell_3/Sigmoid:y:0model/gru_1/zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’a
model/gru_1/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/gru_1/gru_cell_3/subSub%model/gru_1/gru_cell_3/sub/x:output:0"model/gru_1/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’”
model/gru_1/gru_cell_3/mul_2Mulmodel/gru_1/gru_cell_3/sub:z:0)model/gru_1/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
model/gru_1/gru_cell_3/add_3AddV2 model/gru_1/gru_cell_3/mul_1:z:0 model/gru_1/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’z
)model/gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ü
model/gru_1/TensorArrayV2_1TensorListReserve2model/gru_1/TensorArrayV2_1/element_shape:output:0$model/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅR
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
’’’’’’’’’`
model/gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ł
model/gru_1/whileWhile'model/gru_1/while/loop_counter:output:0-model/gru_1/while/maximum_iterations:output:0model/gru_1/time:output:0$model/gru_1/TensorArrayV2_1:handle:0model/gru_1/zeros:output:0$model/gru_1/strided_slice_1:output:0Cmodel/gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0.model_gru_1_gru_cell_3_readvariableop_resource5model_gru_1_gru_cell_3_matmul_readvariableop_resource7model_gru_1_gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *)
body!R
model_gru_1_while_body_386538*)
cond!R
model_gru_1_while_cond_386537*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
<model/gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ē
.model/gru_1/TensorArrayV2Stack/TensorListStackTensorListStackmodel/gru_1/while:output:3Emodel/gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2’’’’’’’’’*
element_dtype0t
!model/gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’m
#model/gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#model/gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
model/gru_1/strided_slice_3StridedSlice7model/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0*model/gru_1/strided_slice_3/stack:output:0,model/gru_1/strided_slice_3/stack_1:output:0,model/gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskq
model/gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
model/gru_1/transpose_1	Transpose7model/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0%model/gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2g
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
value	B :Ģ
model/tf.concat_5/concatConcatV2!model/tf.concat_4/concat:output:0model/gru_1/transpose_1:y:0&model/tf.concat_5/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2q
 model/dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
model/dense_output/ReshapeReshape!model/tf.concat_5/concat:output:0)model/dense_output/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’§
.model/dense_output/dense/MatMul/ReadVariableOpReadVariableOp7model_dense_output_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0ø
model/dense_output/dense/MatMulMatMul#model/dense_output/Reshape:output:06model/dense_output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’¤
/model/dense_output/dense/BiasAdd/ReadVariableOpReadVariableOp8model_dense_output_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Į
 model/dense_output/dense/BiasAddBiasAdd)model/dense_output/dense/MatMul:product:07model/dense_output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
 model/dense_output/dense/SigmoidSigmoid)model/dense_output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’w
"model/dense_output/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’2      °
model/dense_output/Reshape_1Reshape$model/dense_output/dense/Sigmoid:y:0+model/dense_output/Reshape_1/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’2s
"model/dense_output/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ŗ
model/dense_output/Reshape_2Reshape!model/tf.concat_5/concat:output:0+model/dense_output/Reshape_2/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’x
IdentityIdentity%model/dense_output/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2²
NoOpNoOp0^model/dense_output/dense/BiasAdd/ReadVariableOp/^model/dense_output/dense/MatMul/ReadVariableOp+^model/dense_surface/BiasAdd/ReadVariableOp*^model/dense_surface/MatMul/ReadVariableOp+^model/gru/gru_cell_2/MatMul/ReadVariableOp-^model/gru/gru_cell_2/MatMul_1/ReadVariableOp$^model/gru/gru_cell_2/ReadVariableOp^model/gru/while-^model/gru_1/gru_cell_3/MatMul/ReadVariableOp/^model/gru_1/gru_cell_3/MatMul_1/ReadVariableOp&^model/gru_1/gru_cell_3/ReadVariableOp^model/gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 2b
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
:’’’’’’’’’1
%
_user_specified_nameinputs_main:SO
'
_output_shapes
:’’’’’’’’’
$
_user_specified_name
inputs_aux
½

&__inference_model_layer_call_fn_388446
inputs_0
inputs_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:

	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:

	unknown_7:	
	unknown_8:
identity¢StatefulPartitionedCallŅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_388231s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:’’’’’’’’’1
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1
¾
Ŗ
while_cond_390374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390374___redundant_placeholder04
0while_while_cond_390374___redundant_placeholder14
0while_while_cond_390374___redundant_placeholder24
0while_while_cond_390374___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
 O

A__inference_gru_1_layer_call_and_return_conditional_losses_389999
inputs_05
"gru_cell_3_readvariableop_resource:	=
)gru_cell_3_matmul_readvariableop_resource:
?
+gru_cell_3_matmul_1_readvariableop_resource:

identity¢ gru_cell_3/MatMul/ReadVariableOp¢"gru_cell_3/MatMul_1/ReadVariableOp¢gru_cell_3/ReadVariableOp¢whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
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
valueB:Ū
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
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ~
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
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
valueB:ź
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask}
gru_cell_3/ReadVariableOpReadVariableOp"gru_cell_3_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_3/unstackUnpack!gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMulMatMulstrided_slice_2:output:0(gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAddBiasAddgru_cell_3/MatMul:product:0gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
gru_cell_3/splitSplit#gru_cell_3/split/split_dim:output:0gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
"gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_3/MatMul_1MatMulzeros:output:0*gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/BiasAdd_1BiasAddgru_cell_3/MatMul_1:product:0gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’e
gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’g
gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’õ
gru_cell_3/split_1SplitVgru_cell_3/BiasAdd_1:output:0gru_cell_3/Const:output:0%gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_cell_3/addAddV2gru_cell_3/split:output:0gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
gru_cell_3/SigmoidSigmoidgru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/add_1AddV2gru_cell_3/split:output:1gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’h
gru_cell_3/Sigmoid_1Sigmoidgru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_cell_3/mulMulgru_cell_3/Sigmoid_1:y:0gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’{
gru_cell_3/add_2AddV2gru_cell_3/split:output:2gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
gru_cell_3/ReluRelugru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
gru_cell_3/mul_1Mulgru_cell_3/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’U
gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_3/subSubgru_cell_3/sub/x:output:0gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
gru_cell_3/mul_2Mulgru_cell_3/sub:z:0gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’x
gru_cell_3/add_3AddV2gru_cell_3/mul_1:z:0gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ø
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
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
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_3_readvariableop_resource)gru_cell_3_matmul_readvariableop_resource+gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_389910*
condR
while_cond_389909*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ģ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’²
NoOpNoOp!^gru_cell_3/MatMul/ReadVariableOp#^gru_cell_3/MatMul_1/ReadVariableOp^gru_cell_3/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':’’’’’’’’’’’’’’’’’’: : : 2D
 gru_cell_3/MatMul/ReadVariableOp gru_cell_3/MatMul/ReadVariableOp2H
"gru_cell_3/MatMul_1/ReadVariableOp"gru_cell_3/MatMul_1/ReadVariableOp26
gru_cell_3/ReadVariableOpgru_cell_3/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
į

č
model_gru_while_cond_3863730
,model_gru_while_model_gru_while_loop_counter6
2model_gru_while_model_gru_while_maximum_iterations
model_gru_while_placeholder!
model_gru_while_placeholder_1!
model_gru_while_placeholder_22
.model_gru_while_less_model_gru_strided_slice_1H
Dmodel_gru_while_model_gru_while_cond_386373___redundant_placeholder0H
Dmodel_gru_while_model_gru_while_cond_386373___redundant_placeholder1H
Dmodel_gru_while_model_gru_while_cond_386373___redundant_placeholder2H
Dmodel_gru_while_model_gru_while_cond_386373___redundant_placeholder3
model_gru_while_identity

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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::T P
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
°

H__inference_dense_output_layer_call_and_return_conditional_losses_390504

inputs7
$dense_matmul_readvariableop_resource:	3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
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
valueB:Ń
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
valueB"’’’’   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense/Sigmoid:y:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’’’’’’’’’’: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ā

&__inference_dense_layer_call_fn_390747

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallŁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_387350o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Č
§	
A__inference_model_layer_call_and_return_conditional_losses_389112
inputs_0
inputs_19
&gru_gru_cell_2_readvariableop_resource:	@
-gru_gru_cell_2_matmul_readvariableop_resource:	C
/gru_gru_cell_2_matmul_1_readvariableop_resource:
@
,dense_surface_matmul_readvariableop_resource:
<
-dense_surface_biasadd_readvariableop_resource:	;
(gru_1_gru_cell_3_readvariableop_resource:	C
/gru_1_gru_cell_3_matmul_readvariableop_resource:
E
1gru_1_gru_cell_3_matmul_1_readvariableop_resource:
D
1dense_output_dense_matmul_readvariableop_resource:	@
2dense_output_dense_biasadd_readvariableop_resource:
identity¢)dense_output/dense/BiasAdd/ReadVariableOp¢(dense_output/dense/MatMul/ReadVariableOp¢$dense_surface/BiasAdd/ReadVariableOp¢#dense_surface/MatMul/ReadVariableOp¢$gru/gru_cell_2/MatMul/ReadVariableOp¢&gru/gru_cell_2/MatMul_1/ReadVariableOp¢gru/gru_cell_2/ReadVariableOp¢	gru/while¢&gru_1/gru_cell_3/MatMul/ReadVariableOp¢(gru_1/gru_cell_3/MatMul_1/ReadVariableOp¢gru_1/gru_cell_3/ReadVariableOp¢gru_1/whileO
	gru/ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻa
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
valueB:å
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
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
 *    y
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
gru/transpose	Transposeinputs_0gru/transpose/perm:output:0*
T0*+
_output_shapes
:1’’’’’’’’’Z
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
::ķĻc
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
valueB:ļ
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
’’’’’’’’’Ą
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ģ
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅc
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
valueB:ż
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask
gru/gru_cell_2/ReadVariableOpReadVariableOp&gru_gru_cell_2_readvariableop_resource*
_output_shapes
:	*
dtype0
gru/gru_cell_2/unstackUnpack%gru/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp-gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru/gru_cell_2/MatMulMatMulgru/strided_slice_2:output:0,gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/BiasAddBiasAddgru/gru_cell_2/MatMul:product:0gru/gru_cell_2/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ó
gru/gru_cell_2/splitSplit'gru/gru_cell_2/split/split_dim:output:0gru/gru_cell_2/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
&gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp/gru_gru_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru/gru_cell_2/MatMul_1MatMulgru/zeros:output:0.gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/BiasAdd_1BiasAdd!gru/gru_cell_2/MatMul_1:product:0gru/gru_cell_2/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’i
gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’k
 gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
gru/gru_cell_2/split_1SplitV!gru/gru_cell_2/BiasAdd_1:output:0gru/gru_cell_2/Const:output:0)gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru/gru_cell_2/addAddV2gru/gru_cell_2/split:output:0gru/gru_cell_2/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’l
gru/gru_cell_2/SigmoidSigmoidgru/gru_cell_2/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/add_1AddV2gru/gru_cell_2/split:output:1gru/gru_cell_2/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’p
gru/gru_cell_2/Sigmoid_1Sigmoidgru/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/mulMulgru/gru_cell_2/Sigmoid_1:y:0gru/gru_cell_2/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/add_2AddV2gru/gru_cell_2/split:output:2gru/gru_cell_2/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’h
gru/gru_cell_2/ReluRelugru/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
gru/gru_cell_2/mul_1Mulgru/gru_cell_2/Sigmoid:y:0gru/zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/gru_cell_2/subSubgru/gru_cell_2/sub/x:output:0gru/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/mul_2Mulgru/gru_cell_2/sub:z:0!gru/gru_cell_2/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
gru/gru_cell_2/add_3AddV2gru/gru_cell_2/mul_1:z:0gru/gru_cell_2/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ä
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅJ
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
’’’’’’’’’X
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ń
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0&gru_gru_cell_2_readvariableop_resource-gru_gru_cell_2_matmul_readvariableop_resource/gru_gru_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *!
bodyR
gru_while_body_388844*!
condR
gru_while_cond_388843*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ļ
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:1’’’’’’’’’*
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’e
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’1_
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
value	B :
tf.concat_3/concatConcatV2gru/while:output:4inputs_1 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
#dense_surface/MatMul/ReadVariableOpReadVariableOp,dense_surface_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_surface/MatMulMatMultf.concat_3/concat:output:0+dense_surface/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
$dense_surface/BiasAdd/ReadVariableOpReadVariableOp-dense_surface_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0”
dense_surface/BiasAddBiasAdddense_surface/MatMul:product:0,dense_surface/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
dense_surface/ReluReludense_surface/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’o
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’      
tf.reshape_1/ReshapeReshape dense_surface/Relu:activations:0#tf.reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’Y
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :“
tf.concat_4/concatConcatV2gru/transpose_1:y:0tf.reshape_1/Reshape:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2d
gru_1/ShapeShapetf.concat_4/concat:output:0*
T0*
_output_shapes
::ķĻc
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
valueB:ļ
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
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
 *    
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_1/transpose	Transposetf.concat_4/concat:output:0gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:2’’’’’’’’’^
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
::ķĻe
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
valueB:ł
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
’’’’’’’’’Ę
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ^
gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
gru_1/ReverseV2	ReverseV2gru_1/transpose:y:0gru_1/ReverseV2/axis:output:0*
T0*,
_output_shapes
:2’’’’’’’’’
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ÷
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/ReverseV2:output:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅe
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
valueB:
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask
gru_1/gru_cell_3/ReadVariableOpReadVariableOp(gru_1_gru_cell_3_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_1/gru_cell_3/unstackUnpack'gru_1/gru_cell_3/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&gru_1/gru_cell_3/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¤
gru_1/gru_cell_3/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/BiasAddBiasAdd!gru_1/gru_cell_3/MatMul:product:0!gru_1/gru_cell_3/unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
 gru_1/gru_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
gru_1/gru_cell_3/splitSplit)gru_1/gru_cell_3/split/split_dim:output:0!gru_1/gru_cell_3/BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
(gru_1/gru_cell_3/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_1/gru_cell_3/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ 
gru_1/gru_cell_3/BiasAdd_1BiasAdd#gru_1/gru_cell_3/MatMul_1:product:0!gru_1/gru_cell_3/unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’k
gru_1/gru_cell_3/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’m
"gru_1/gru_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
gru_1/gru_cell_3/split_1SplitV#gru_1/gru_cell_3/BiasAdd_1:output:0gru_1/gru_cell_3/Const:output:0+gru_1/gru_cell_3/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split
gru_1/gru_cell_3/addAddV2gru_1/gru_cell_3/split:output:0!gru_1/gru_cell_3/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
gru_1/gru_cell_3/SigmoidSigmoidgru_1/gru_cell_3/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/add_1AddV2gru_1/gru_cell_3/split:output:1!gru_1/gru_cell_3/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’t
gru_1/gru_cell_3/Sigmoid_1Sigmoidgru_1/gru_cell_3/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/mulMulgru_1/gru_cell_3/Sigmoid_1:y:0!gru_1/gru_cell_3/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/add_2AddV2gru_1/gru_cell_3/split:output:2gru_1/gru_cell_3/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
gru_1/gru_cell_3/ReluRelugru_1/gru_cell_3/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/mul_1Mulgru_1/gru_cell_3/Sigmoid:y:0gru_1/zeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’[
gru_1/gru_cell_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/gru_cell_3/subSubgru_1/gru_cell_3/sub/x:output:0gru_1/gru_cell_3/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/mul_2Mulgru_1/gru_cell_3/sub:z:0#gru_1/gru_cell_3/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’
gru_1/gru_cell_3/add_3AddV2gru_1/gru_cell_3/mul_1:z:0gru_1/gru_cell_3/mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ź
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅL

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
’’’’’’’’’Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_3_readvariableop_resource/gru_1_gru_cell_3_matmul_readvariableop_resource1gru_1_gru_cell_3_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_1_while_body_389008*#
condR
gru_1_while_cond_389007*9
output_shapes(
&: : : : :’’’’’’’’’: : : : : *
parallel_iterations 
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Õ
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2’’’’’’’’’*
element_dtype0n
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’g
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2a
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
value	B :“
tf.concat_5/concatConcatV2tf.concat_4/concat:output:0gru_1/transpose_1:y:0 tf.concat_5/concat/axis:output:0*
N*
T0*,
_output_shapes
:’’’’’’’’’2k
dense_output/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
dense_output/ReshapeReshapetf.concat_5/concat:output:0#dense_output/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’
(dense_output/dense/MatMul/ReadVariableOpReadVariableOp1dense_output_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¦
dense_output/dense/MatMulMatMuldense_output/Reshape:output:00dense_output/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
)dense_output/dense/BiasAdd/ReadVariableOpReadVariableOp2dense_output_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Æ
dense_output/dense/BiasAddBiasAdd#dense_output/dense/MatMul:product:01dense_output/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’|
dense_output/dense/SigmoidSigmoid#dense_output/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’q
dense_output/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’2      
dense_output/Reshape_1Reshapedense_output/dense/Sigmoid:y:0%dense_output/Reshape_1/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’2m
dense_output/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
dense_output/Reshape_2Reshapetf.concat_5/concat:output:0%dense_output/Reshape_2/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’r
IdentityIdentitydense_output/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’2ź
NoOpNoOp*^dense_output/dense/BiasAdd/ReadVariableOp)^dense_output/dense/MatMul/ReadVariableOp%^dense_surface/BiasAdd/ReadVariableOp$^dense_surface/MatMul/ReadVariableOp%^gru/gru_cell_2/MatMul/ReadVariableOp'^gru/gru_cell_2/MatMul_1/ReadVariableOp^gru/gru_cell_2/ReadVariableOp
^gru/while'^gru_1/gru_cell_3/MatMul/ReadVariableOp)^gru_1/gru_cell_3/MatMul_1/ReadVariableOp ^gru_1/gru_cell_3/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:’’’’’’’’’1:’’’’’’’’’: : : : : : : : : : 2V
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
:’’’’’’’’’1
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1
¾
Ŗ
while_cond_390064
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390064___redundant_placeholder04
0while_while_cond_390064___redundant_placeholder14
0while_while_cond_390064___redundant_placeholder24
0while_while_cond_390064___redundant_placeholder3
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
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :’’’’’’’’’: :::::J F
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
: :.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
:
Ø
Ū
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_390593

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ’’’’\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’É
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:’’’’’’’’’J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0"ó
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultī
A

inputs_aux3
serving_default_inputs_aux:0’’’’’’’’’
G
inputs_main8
serving_default_inputs_main:0’’’’’’’’’1D
dense_output4
StatefulPartitionedCall:0’’’’’’’’’2tensorflow/serving/predict:ō
×
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
Ś
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
»
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
Ś
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
°
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
Ź
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
Ć
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32Ų
&__inference_model_layer_call_fn_388188
&__inference_model_layer_call_fn_388254
&__inference_model_layer_call_fn_388420
&__inference_model_layer_call_fn_388446µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
Æ
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32Ä
A__inference_model_layer_call_and_return_conditional_losses_387772
A__inference_model_layer_call_and_return_conditional_losses_388121
A__inference_model_layer_call_and_return_conditional_losses_388779
A__inference_model_layer_call_and_return_conditional_losses_389112µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
ÜBŁ
!__inference__wrapped_model_386642inputs_main
inputs_aux"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
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
¹

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
Š
Ztrace_0
[trace_1
\trace_2
]trace_32å
$__inference_gru_layer_call_fn_389125
$__inference_gru_layer_call_fn_389138
$__inference_gru_layer_call_fn_389151
$__inference_gru_layer_call_fn_389164Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zZtrace_0z[trace_1z\trace_2z]trace_3
¼
^trace_0
_trace_1
`trace_2
atrace_32Ń
?__inference_gru_layer_call_and_return_conditional_losses_389318
?__inference_gru_layer_call_and_return_conditional_losses_389472
?__inference_gru_layer_call_and_return_conditional_losses_389626
?__inference_gru_layer_call_and_return_conditional_losses_389780Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z^trace_0z_trace_1z`trace_2zatrace_3
"
_generic_user_object
č
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
­
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
č
ntrace_02Ė
.__inference_dense_surface_layer_call_fn_389789
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zntrace_0

otrace_02ę
I__inference_dense_surface_layer_call_and_return_conditional_losses_389800
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zotrace_0
(:&
2dense_surface/kernel
!:2dense_surface/bias
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
¹

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
Ų
vtrace_0
wtrace_1
xtrace_2
ytrace_32ķ
&__inference_gru_1_layer_call_fn_389811
&__inference_gru_1_layer_call_fn_389822
&__inference_gru_1_layer_call_fn_389833
&__inference_gru_1_layer_call_fn_389844Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zvtrace_0zwtrace_1zxtrace_2zytrace_3
Ä
ztrace_0
{trace_1
|trace_2
}trace_32Ł
A__inference_gru_1_layer_call_and_return_conditional_losses_389999
A__inference_gru_1_layer_call_and_return_conditional_losses_390154
A__inference_gru_1_layer_call_and_return_conditional_losses_390309
A__inference_gru_1_layer_call_and_return_conditional_losses_390464Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zztrace_0z{trace_1z|trace_2z}trace_3
"
_generic_user_object
ķ
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
Ń
trace_0
trace_12
-__inference_dense_output_layer_call_fn_390473
-__inference_dense_output_layer_call_fn_390482µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1

trace_0
trace_12Ģ
H__inference_dense_output_layer_call_and_return_conditional_losses_390504
H__inference_dense_output_layer_call_and_return_conditional_losses_390526µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1
Į
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
(:&	2gru/gru_cell_2/kernel
3:1
2gru/gru_cell_2/recurrent_kernel
&:$	2gru/gru_cell_2/bias
+:)
2gru_1/gru_cell_3/kernel
5:3
2!gru_1/gru_cell_3/recurrent_kernel
(:&	2gru_1/gru_cell_3/bias
&:$	2dense_output/kernel
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
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
žBū
&__inference_model_layer_call_fn_388188inputs_main
inputs_aux"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
žBū
&__inference_model_layer_call_fn_388254inputs_main
inputs_aux"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
&__inference_model_layer_call_fn_388420inputs_0inputs_1"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
&__inference_model_layer_call_fn_388446inputs_0inputs_1"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_387772inputs_main
inputs_aux"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_388121inputs_main
inputs_aux"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_388779inputs_0inputs_1"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_389112inputs_0inputs_1"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
'
P0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
µ2²Æ
¦²¢
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
ŁBÖ
$__inference_signature_wrapper_388394
inputs_auxinputs_main"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
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
B’
$__inference_gru_layer_call_fn_389125inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B’
$__inference_gru_layer_call_fn_389138inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
$__inference_gru_layer_call_fn_389151inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
$__inference_gru_layer_call_fn_389164inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
?__inference_gru_layer_call_and_return_conditional_losses_389318inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
?__inference_gru_layer_call_and_return_conditional_losses_389472inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
?__inference_gru_layer_call_and_return_conditional_losses_389626inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
?__inference_gru_layer_call_and_return_conditional_losses_389780inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
Ė
trace_0
trace_12
+__inference_gru_cell_2_layer_call_fn_390540
+__inference_gru_cell_2_layer_call_fn_390554³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1

trace_0
trace_12Ę
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_390593
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_390632³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1
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
ŲBÕ
.__inference_dense_surface_layer_call_fn_389789inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
óBš
I__inference_dense_surface_layer_call_and_return_conditional_losses_389800inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
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
B
&__inference_gru_1_layer_call_fn_389811inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
&__inference_gru_1_layer_call_fn_389822inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B’
&__inference_gru_1_layer_call_fn_389833inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B’
&__inference_gru_1_layer_call_fn_389844inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_gru_1_layer_call_and_return_conditional_losses_389999inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_gru_1_layer_call_and_return_conditional_losses_390154inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_gru_1_layer_call_and_return_conditional_losses_390309inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_gru_1_layer_call_and_return_conditional_losses_390464inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
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
¶
 non_trainable_variables
”layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ė
„trace_0
¦trace_12
+__inference_gru_cell_3_layer_call_fn_390646
+__inference_gru_cell_3_layer_call_fn_390660³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z„trace_0z¦trace_1

§trace_0
Øtrace_12Ę
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_390699
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_390738³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z§trace_0zØtrace_1
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
ōBń
-__inference_dense_output_layer_call_fn_390473inputs"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ōBń
-__inference_dense_output_layer_call_fn_390482inputs"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
H__inference_dense_output_layer_call_and_return_conditional_losses_390504inputs"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
H__inference_dense_output_layer_call_and_return_conditional_losses_390526inputs"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
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
ø
©non_trainable_variables
Ŗlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ā
®trace_02Ć
&__inference_dense_layer_call_fn_390747
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z®trace_0
ż
Ætrace_02Ž
A__inference_dense_layer_call_and_return_conditional_losses_390758
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zÆtrace_0
R
°	variables
±	keras_api

²total

³count"
_tf_keras_metric
c
“	variables
µ	keras_api

¶total

·count
ø
_fn_kwargs"
_tf_keras_metric
c
¹	variables
ŗ	keras_api

»total

¼count
½
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
śB÷
+__inference_gru_cell_2_layer_call_fn_390540inputsstates_0"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
śB÷
+__inference_gru_cell_2_layer_call_fn_390554inputsstates_0"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_390593inputsstates_0"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_390632inputsstates_0"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
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
śB÷
+__inference_gru_cell_3_layer_call_fn_390646inputsstates_0"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
śB÷
+__inference_gru_cell_3_layer_call_fn_390660inputsstates_0"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_390699inputsstates_0"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_390738inputsstates_0"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
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
ŠBĶ
&__inference_dense_layer_call_fn_390747inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ėBč
A__inference_dense_layer_call_and_return_conditional_losses_390758inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
0
²0
³1"
trackable_list_wrapper
.
°	variables"
_generic_user_object
:  (2total
:  (2count
0
¶0
·1"
trackable_list_wrapper
.
“	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
»0
¼1"
trackable_list_wrapper
.
¹	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperŲ
!__inference__wrapped_model_386642²
<:;%&?=>@Ac¢`
Y¢V
TQ
)&
inputs_main’’’’’’’’’1
$!

inputs_aux’’’’’’’’’
Ŗ "?Ŗ<
:
dense_output*'
dense_output’’’’’’’’’2©
A__inference_dense_layer_call_and_return_conditional_losses_390758d@A0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ ",¢)
"
tensor_0’’’’’’’’’
 
&__inference_dense_layer_call_fn_390747Y@A0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "!
unknown’’’’’’’’’Ó
H__inference_dense_output_layer_call_and_return_conditional_losses_390504@AE¢B
;¢8
.+
inputs’’’’’’’’’’’’’’’’’’
p

 
Ŗ "9¢6
/,
tensor_0’’’’’’’’’’’’’’’’’’
 Ó
H__inference_dense_output_layer_call_and_return_conditional_losses_390526@AE¢B
;¢8
.+
inputs’’’’’’’’’’’’’’’’’’
p 

 
Ŗ "9¢6
/,
tensor_0’’’’’’’’’’’’’’’’’’
 ¬
-__inference_dense_output_layer_call_fn_390473{@AE¢B
;¢8
.+
inputs’’’’’’’’’’’’’’’’’’
p

 
Ŗ ".+
unknown’’’’’’’’’’’’’’’’’’¬
-__inference_dense_output_layer_call_fn_390482{@AE¢B
;¢8
.+
inputs’’’’’’’’’’’’’’’’’’
p 

 
Ŗ ".+
unknown’’’’’’’’’’’’’’’’’’²
I__inference_dense_surface_layer_call_and_return_conditional_losses_389800e%&0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 
.__inference_dense_surface_layer_call_fn_389789Z%&0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ ""
unknown’’’’’’’’’Ł
A__inference_gru_1_layer_call_and_return_conditional_losses_389999?=>P¢M
F¢C
52
0-
inputs_0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ ":¢7
0-
tensor_0’’’’’’’’’’’’’’’’’’
 Ł
A__inference_gru_1_layer_call_and_return_conditional_losses_390154?=>P¢M
F¢C
52
0-
inputs_0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ ":¢7
0-
tensor_0’’’’’’’’’’’’’’’’’’
 æ
A__inference_gru_1_layer_call_and_return_conditional_losses_390309z?=>@¢=
6¢3
%"
inputs’’’’’’’’’2

 
p

 
Ŗ "1¢.
'$
tensor_0’’’’’’’’’2
 æ
A__inference_gru_1_layer_call_and_return_conditional_losses_390464z?=>@¢=
6¢3
%"
inputs’’’’’’’’’2

 
p 

 
Ŗ "1¢.
'$
tensor_0’’’’’’’’’2
 ³
&__inference_gru_1_layer_call_fn_389811?=>P¢M
F¢C
52
0-
inputs_0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ "/,
unknown’’’’’’’’’’’’’’’’’’³
&__inference_gru_1_layer_call_fn_389822?=>P¢M
F¢C
52
0-
inputs_0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ "/,
unknown’’’’’’’’’’’’’’’’’’
&__inference_gru_1_layer_call_fn_389833o?=>@¢=
6¢3
%"
inputs’’’’’’’’’2

 
p

 
Ŗ "&#
unknown’’’’’’’’’2
&__inference_gru_1_layer_call_fn_389844o?=>@¢=
6¢3
%"
inputs’’’’’’’’’2

 
p 

 
Ŗ "&#
unknown’’’’’’’’’2
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_390593Č<:;]¢Z
S¢P
 
inputs’’’’’’’’’
(¢%
# 
states_0’’’’’’’’’
p
Ŗ "b¢_
X¢U
%"

tensor_0_0’’’’’’’’’
,)
'$
tensor_0_1_0’’’’’’’’’
 
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_390632Č<:;]¢Z
S¢P
 
inputs’’’’’’’’’
(¢%
# 
states_0’’’’’’’’’
p 
Ŗ "b¢_
X¢U
%"

tensor_0_0’’’’’’’’’
,)
'$
tensor_0_1_0’’’’’’’’’
 ź
+__inference_gru_cell_2_layer_call_fn_390540ŗ<:;]¢Z
S¢P
 
inputs’’’’’’’’’
(¢%
# 
states_0’’’’’’’’’
p
Ŗ "T¢Q
# 
tensor_0’’’’’’’’’
*'
%"

tensor_1_0’’’’’’’’’ź
+__inference_gru_cell_2_layer_call_fn_390554ŗ<:;]¢Z
S¢P
 
inputs’’’’’’’’’
(¢%
# 
states_0’’’’’’’’’
p 
Ŗ "T¢Q
# 
tensor_0’’’’’’’’’
*'
%"

tensor_1_0’’’’’’’’’
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_390699É?=>^¢[
T¢Q
!
inputs’’’’’’’’’
(¢%
# 
states_0’’’’’’’’’
p
Ŗ "b¢_
X¢U
%"

tensor_0_0’’’’’’’’’
,)
'$
tensor_0_1_0’’’’’’’’’
 
F__inference_gru_cell_3_layer_call_and_return_conditional_losses_390738É?=>^¢[
T¢Q
!
inputs’’’’’’’’’
(¢%
# 
states_0’’’’’’’’’
p 
Ŗ "b¢_
X¢U
%"

tensor_0_0’’’’’’’’’
,)
'$
tensor_0_1_0’’’’’’’’’
 ė
+__inference_gru_cell_3_layer_call_fn_390646»?=>^¢[
T¢Q
!
inputs’’’’’’’’’
(¢%
# 
states_0’’’’’’’’’
p
Ŗ "T¢Q
# 
tensor_0’’’’’’’’’
*'
%"

tensor_1_0’’’’’’’’’ė
+__inference_gru_cell_3_layer_call_fn_390660»?=>^¢[
T¢Q
!
inputs’’’’’’’’’
(¢%
# 
states_0’’’’’’’’’
p 
Ŗ "T¢Q
# 
tensor_0’’’’’’’’’
*'
%"

tensor_1_0’’’’’’’’’
?__inference_gru_layer_call_and_return_conditional_losses_389318Ą<:;O¢L
E¢B
41
/,
inputs_0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ "h¢e
^[
2/

tensor_0_0’’’’’’’’’’’’’’’’’’
%"

tensor_0_1’’’’’’’’’
 
?__inference_gru_layer_call_and_return_conditional_losses_389472Ą<:;O¢L
E¢B
41
/,
inputs_0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ "h¢e
^[
2/

tensor_0_0’’’’’’’’’’’’’’’’’’
%"

tensor_0_1’’’’’’’’’
 ė
?__inference_gru_layer_call_and_return_conditional_losses_389626§<:;?¢<
5¢2
$!
inputs’’’’’’’’’1

 
p

 
Ŗ "_¢\
UR
)&

tensor_0_0’’’’’’’’’1
%"

tensor_0_1’’’’’’’’’
 ė
?__inference_gru_layer_call_and_return_conditional_losses_389780§<:;?¢<
5¢2
$!
inputs’’’’’’’’’1

 
p 

 
Ŗ "_¢\
UR
)&

tensor_0_0’’’’’’’’’1
%"

tensor_0_1’’’’’’’’’
 Ū
$__inference_gru_layer_call_fn_389125²<:;O¢L
E¢B
41
/,
inputs_0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ "ZW
0-
tensor_0’’’’’’’’’’’’’’’’’’
# 
tensor_1’’’’’’’’’Ū
$__inference_gru_layer_call_fn_389138²<:;O¢L
E¢B
41
/,
inputs_0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ "ZW
0-
tensor_0’’’’’’’’’’’’’’’’’’
# 
tensor_1’’’’’’’’’Ā
$__inference_gru_layer_call_fn_389151<:;?¢<
5¢2
$!
inputs’’’’’’’’’1

 
p

 
Ŗ "QN
'$
tensor_0’’’’’’’’’1
# 
tensor_1’’’’’’’’’Ā
$__inference_gru_layer_call_fn_389164<:;?¢<
5¢2
$!
inputs’’’’’’’’’1

 
p 

 
Ŗ "QN
'$
tensor_0’’’’’’’’’1
# 
tensor_1’’’’’’’’’ń
A__inference_model_layer_call_and_return_conditional_losses_387772«
<:;%&?=>@Ak¢h
a¢^
TQ
)&
inputs_main’’’’’’’’’1
$!

inputs_aux’’’’’’’’’
p

 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’2
 ń
A__inference_model_layer_call_and_return_conditional_losses_388121«
<:;%&?=>@Ak¢h
a¢^
TQ
)&
inputs_main’’’’’’’’’1
$!

inputs_aux’’’’’’’’’
p 

 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’2
 ģ
A__inference_model_layer_call_and_return_conditional_losses_388779¦
<:;%&?=>@Af¢c
\¢Y
OL
&#
inputs_0’’’’’’’’’1
"
inputs_1’’’’’’’’’
p

 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’2
 ģ
A__inference_model_layer_call_and_return_conditional_losses_389112¦
<:;%&?=>@Af¢c
\¢Y
OL
&#
inputs_0’’’’’’’’’1
"
inputs_1’’’’’’’’’
p 

 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’2
 Ė
&__inference_model_layer_call_fn_388188 
<:;%&?=>@Ak¢h
a¢^
TQ
)&
inputs_main’’’’’’’’’1
$!

inputs_aux’’’’’’’’’
p

 
Ŗ "%"
unknown’’’’’’’’’2Ė
&__inference_model_layer_call_fn_388254 
<:;%&?=>@Ak¢h
a¢^
TQ
)&
inputs_main’’’’’’’’’1
$!

inputs_aux’’’’’’’’’
p 

 
Ŗ "%"
unknown’’’’’’’’’2Ę
&__inference_model_layer_call_fn_388420
<:;%&?=>@Af¢c
\¢Y
OL
&#
inputs_0’’’’’’’’’1
"
inputs_1’’’’’’’’’
p

 
Ŗ "%"
unknown’’’’’’’’’2Ę
&__inference_model_layer_call_fn_388446
<:;%&?=>@Af¢c
\¢Y
OL
&#
inputs_0’’’’’’’’’1
"
inputs_1’’’’’’’’’
p 

 
Ŗ "%"
unknown’’’’’’’’’2ó
$__inference_signature_wrapper_388394Ź
<:;%&?=>@A{¢x
¢ 
qŖn
2

inputs_aux$!

inputs_aux’’’’’’’’’
8
inputs_main)&
inputs_main’’’’’’’’’1"?Ŗ<
:
dense_output*'
dense_output’’’’’’’’’2