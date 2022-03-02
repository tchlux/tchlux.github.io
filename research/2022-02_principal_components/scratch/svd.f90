! Orthogonalize the column vectors in A.
SUBROUTINE ORTHOGONALIZE(A, MAGNITUDES, RANK, TRANS, PIVOT)
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  IMPLICIT NONE
  REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
  REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: MAGNITUDES
  INTEGER, INTENT(OUT), OPTIONAL :: RANK
  LOGICAL, INTENT(IN), OPTIONAL :: TRANS, PIVOT
  ! Local variables.
  INTEGER :: I, J
  REAL(KIND=RT), DIMENSION(:), ALLOCATABLE :: MULTIPLIERS, VEC
  REAL(KIND=RT) :: LENGTH
  LOGICAL :: T, P
  ! Set the default for whether to orthogonalize the matirx or its transpose.
  IF (PRESENT(TRANS)) THEN
     T = TRANS
  ELSE
     IF (SIZE(A,1) .LE. SIZE(A,2)) THEN
        T = .FALSE.
     ELSE
        T = .TRUE.
     END IF
  END IF
  ! Set the default for pivoting.
  IF (PRESENT(PIVOT)) THEN
     P = PIVOT
  ELSE
     P = .TRUE.
  END IF
  ! Set the initial value for the rank of the matrix.
  IF (PRESENT(RANK)) RANK = 0
  ! NEAR DUPLICATION OF THE SAME CODE FOR TRANSPOSE AND NORMAL OPERATION.
  IF (.NOT. T) THEN
     ALLOCATE(MULTIPLIERS(SIZE(A,2)), VEC(SIZE(A,1)))
     ! Iterate over the vectors, orthogonalizing, normalizing, and pivoting.
     iterative_col_orthgonalization : DO I = 1, SIZE(A,2)
        ! Swap the largest magnitude vector to the front.
        IF (P) THEN
           MAGNITUDES(I:) = SUM(A(:,I:)**2, 1)
           J = I-1 + MAXLOC(MAGNITUDES(I:), 1)
           ! Exit early if all the remaining vectors have zero magnitude.
           IF (MAGNITUDES(J) .LE. EPSILON(1.0_RT)) THEN
              MAGNITUDES(I:) = 0.0_RT
              A(:,I:) = 0.0_RT
              EXIT iterative_col_orthgonalization
           ELSE IF (J .NE. I) THEN
              LENGTH = MAGNITUDES(I)
              MAGNITUDES(I) = MAGNITUDES(J)
              MAGNITUDES(J) = LENGTH
              VEC(:) = A(:,I)
              A(:,I) = A(:,J)
              A(:,J) = VEC(:)
           END IF
           IF (PRESENT(RANK)) RANK = RANK + 1
        ELSE
           MAGNITUDES(I) = SUM(A(:,I)**2)
           IF (MAGNITUDES(I) .LE. EPSILON(1.0_RT)) THEN
              MAGNITUDES(I) = 0.0_RT
              A(:,I) = 0.0_RT
              CYCLE iterative_col_orthgonalization
           END IF
        END IF
        ! Normalize the length of the vector.
        MAGNITUDES(I) = SQRT(MAGNITUDES(I))
        A(:,I) = A(:,I) / MAGNITUDES(I)
        ! Subtract the component in the direction of the first from all others.
        IF (I .LT. SIZE(A,2)) THEN
           MULTIPLIERS(I+1:) = MATMUL(A(:,I), A(:,I+1:))
           DO J = I+1, SIZE(A,2)
              A(:,J) = A(:,J) - MULTIPLIERS(J) * A(:,I)
           END DO
        END IF
     END DO iterative_col_orthgonalization
  ! NEAR DUPLICATION OF THE SAME CODE FOR TRANSPOSE AND NORMAL OPERATION.
  ELSE
     ALLOCATE(MULTIPLIERS(SIZE(A,1)), VEC(SIZE(A,2)))
     ! Iterate over the vectors, orthogonalizing, normalizing, and pivoting.
     iterative_row_orthgonalization : DO I = 1, SIZE(A,1)
        ! Swap the largest magnitude vector to the front.
        IF (P) THEN
           MAGNITUDES(I:) = SUM(A(I:,:)**2, 2)
           J = I-1 + MAXLOC(MAGNITUDES(I:), 1)
           ! Exit early if all the remaining vectors have zero magnitude.
           IF (MAGNITUDES(J) .LE. EPSILON(1.0_RT)) THEN
              MAGNITUDES(I:) = 0.0_RT
              A(I:,:) = 0.0_RT
              EXIT iterative_row_orthgonalization
           ELSE IF (J .NE. I) THEN
              LENGTH = MAGNITUDES(I)
              MAGNITUDES(I) = MAGNITUDES(J)
              MAGNITUDES(J) = LENGTH
              VEC(:) = A(I,:)
              A(I,:) = A(J,:)
              A(J,:) = VEC(:)
           END IF
           IF (PRESENT(RANK)) RANK = RANK + 1
        ELSE
           MAGNITUDES(I) = SUM(A(I,:)**2)
           IF (MAGNITUDES(I) .LE. EPSILON(1.0_RT)) THEN
              MAGNITUDES(I) = 0.0_RT
              A(I,:) = 0.0_RT
              CYCLE iterative_row_orthgonalization
           END IF
        END IF
        ! Normalize the length of the vector.
        MAGNITUDES(I) = SQRT(MAGNITUDES(I))
        A(I,:) = A(I,:) / MAGNITUDES(I)
        ! Subtract the component in the direction of the first from all others.
        IF (I .LT. SIZE(A,1)) THEN
           MULTIPLIERS(I+1:) = MATMUL(A(I+1:,:), A(I,:))
           DO J = I+1, SIZE(A,1)
              A(J,:) = A(J,:) - MULTIPLIERS(J) * A(I,:)
           END DO
        END IF
     END DO iterative_row_orthgonalization
  END IF
  DEALLOCATE(MULTIPLIERS, VEC)
END SUBROUTINE ORTHOGONALIZE


! Generate randomly distributed vectors on the sphere.
SUBROUTINE RANDOM_UNIT_VECTORS(COLUMN_VECTORS)
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  IMPLICIT NONE
  REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: COLUMN_VECTORS
  ! Local variables.
  REAL(KIND=RT), DIMENSION(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2)) :: TEMP_VECS
  REAL(KIND=RT), PARAMETER :: PI = 3.141592653589793
  INTEGER :: I, J
  INTERFACE
     SUBROUTINE ORTHOGONALIZE(A, MAGNITUDES, RANK, TRANS, PIVOT)
       USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
       REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
       REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: MAGNITUDES
       INTEGER, INTENT(OUT), OPTIONAL :: RANK
       LOGICAL, INTENT(IN), OPTIONAL :: TRANS, PIVOT
     END SUBROUTINE ORTHOGONALIZE
  END INTERFACE
  ! Skip empty vector sets.
  IF (SIZE(COLUMN_VECTORS) .LE. 0) RETURN
  ! Generate random numbers in the range [0,1].
  CALL RANDOM_NUMBER(COLUMN_VECTORS(:,:))
  CALL RANDOM_NUMBER(TEMP_VECS(:,:))
  ! Map the random uniform numbers to a normal distribution.
  COLUMN_VECTORS(:,:) = SQRT(-LOG(COLUMN_VECTORS(:,:))) * COS(PI * TEMP_VECS(:,:))
  ! Make the vectors uniformly distributed on the unit ball (for dimension > 1).
  IF (SIZE(COLUMN_VECTORS,1) .GT. 1) THEN
     ! Normalize all vectors to have unit length.
     DO I = 1, SIZE(COLUMN_VECTORS,2)
        COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / NORM2(COLUMN_VECTORS(:,I))
     END DO
  END IF
  ! Orthonormalize the first components of the column
  !  vectors to ensure those are well spaced.
  I = MIN(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2))
  IF (I .GT. 1) CALL ORTHOGONALIZE(COLUMN_VECTORS(:,1:I), TEMP_VECS(1,:))
END SUBROUTINE RANDOM_UNIT_VECTORS


! Compute the full singular value decomposition.
RECURSIVE SUBROUTINE SVD(A, U, S, VT, KMAX, STEPS)
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  IMPLICIT NONE
  REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: A
  REAL(KIND=RT), INTENT(OUT), DIMENSION(MAX(SIZE(A,1),SIZE(A,2)),MIN(SIZE(A,1),SIZE(A,2))), OPTIONAL :: U
  REAL(KIND=RT), INTENT(OUT), DIMENSION(MIN(SIZE(A,1),SIZE(A,2))) :: S
  REAL(KIND=RT), INTENT(OUT), DIMENSION(MIN(SIZE(A,1),SIZE(A,2)),MIN(SIZE(A,1),SIZE(A,2))), OPTIONAL :: VT
  INTEGER, INTENT(IN), OPTIONAL :: KMAX, STEPS
  ! Local variables.
  REAL(KIND=RT), DIMENSION(MAX(SIZE(A,1),SIZE(A,2))) :: MAGS
  REAL(KIND=RT), ALLOCATABLE, DIMENSION(:,:) :: AAT, Q, QTEMP, PROJECTION
  INTEGER :: N, M, I, J, K, NUM_STEPS, RANK
  REAL(KIND=RT) :: MULTIPLIER
  EXTERNAL :: SGEMM, SSYRK
  INTERFACE
     SUBROUTINE ORTHOGONALIZE(A, MAGNITUDES, RANK, TRANS, PIVOT)
       USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
       REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
       REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: MAGNITUDES
       INTEGER, INTENT(OUT), OPTIONAL :: RANK
       LOGICAL, INTENT(IN), OPTIONAL :: TRANS, PIVOT
     END SUBROUTINE ORTHOGONALIZE
     SUBROUTINE RANDOM_UNIT_VECTORS(COLUMN_VECTORS)
       USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
       REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: COLUMN_VECTORS
     END SUBROUTINE RANDOM_UNIT_VECTORS
  END INTERFACE
  ! Set the number of steps.
  IF (PRESENT(STEPS)) THEN
     NUM_STEPS = STEPS
  ELSE
     NUM_STEPS = 0
  END IF
  ! Set the number of vectors.
  IF (PRESENT(KMAX)) THEN
     K = MIN(MAX(1,KMAX), MIN(SIZE(A,1), SIZE(A,2)))
  ELSE
     K = MIN(SIZE(A,1), SIZE(A,2))
  END IF
  ! If the number of singular values is reduced, construct a projection.
  IF (K .LT. MIN(SIZE(A,1),SIZE(A,2))) THEN
     PRINT *, 'ERROR: Reduced SVD (k < min(m,n)) is not implemented.'
     K = MIN(SIZE(A,1), SIZE(A,2))
     ! - Must project the data, before or after A.At?
     ! - Must project result to higher dimension as well.
     ! 
     ! ALLOCATE(PROJECTION(1:MIN(SIZE(A,1),SIZE(A,2)),1:K))
     ! CALL RANDOM_UNIT_VECTORS(PROJECTION(:,:))
     ! IF (SIZE(A,1) .LE. SIZE(A,2)) THEN
     !    A(:K,:) = MATMUL(TRANSPOSE(PROJECTION(:,:)), A(:,:))
     ! ELSE
     !    A(:,:K) = MATMUL(A(:,:), PROJECTION(:,:))
     ! END IF
  END IF
  ! Find the multiplier on A.
  MULTIPLIER = MAXVAL(ABS(A(:,:)))
  IF (MULTIPLIER .EQ. 0.0_RT) THEN
     S(:) = 0.0_RT
     IF (PRESENT(VT)) VT(:,:) = 0.0_RT
     IF (PRESENT(U)) U(:,:) = 0.0_RT
     RETURN
  END IF
  MULTIPLIER = 1.0_RT / MULTIPLIER
  ! Allocate VTEMP and AAT.
  ALLOCATE(AAT(1:K,1:K), Q(1:K,1:K), QTEMP(1:K,1:K))
  ! Compute AAT.
  IF (SIZE(A,2) .LE. SIZE(A,1)) THEN
     ! AAT(:,:) = MATMUL(TRANSPOSE(A(:,:)), A(:,:))
     CALL SSYRK('U', 'T', K, SIZE(A,1), MULTIPLIER**2, A(:,:K), &
          SIZE(A,1), 0.0_RT, AAT(:,:), K)
  ELSE
     ! AAT(:,:) = MATMUL(A(:,:), TRANSPOSE(A(:,:)))
     CALL SSYRK('U', 'N', K, SIZE(A,2), MULTIPLIER**2, A(:K,:), &
          K, 0.0_RT, AAT(:,:), K)
  END IF
  ! Copy the upper diagnoal portion into the lower diagonal portion.
  DO I = 1, K
     AAT(I+1:,I) = AAT(I,I+1:)
  END DO
  ! Compute initial right singular vectors.
  Q(:,:) = AAT(:,:)
  CALL ORTHOGONALIZE(Q(:,:), S(:), RANK)
  ! Do power iterations.
  power_iteration : DO I = 1, NUM_STEPS
     QTEMP(:,:) = Q(:,:)
     ! Q(:,:) = MATMUL(TRANSPOSE(AAT(:,:)), QTEMP(:,:))
     CALL SGEMM('N', 'N', K, K, K, 1.0_RT, &
          AAT(:,:), K, QTEMP(:,:), K, 0.0_RT, &
          Q(:,:), K)
     CALL ORTHOGONALIZE(Q(:,:), S(:), RANK)
     IF (SUM((QTEMP(:,:) - Q(:,:))**2) .LT. K*SQRT(EPSILON(1.0_RT))) THEN
        EXIT power_iteration
     END IF
  END DO power_iteration
  ! Compute the singular values.
  WHERE (S(:) .NE. 0.0_RT)
     S(:) = SQRT(S(:))
  END WHERE
  S(:) = S(:) / MULTIPLIER
  ! Make sure Q is fully orthonormal.
  IF (PRESENT(VT) .OR. PRESENT(U)) THEN
     IF (RANK .LT. SIZE(S)) THEN
        CALL ORTHOGONALIZE(Q(:,:), MAGS(:K), TRANS=.TRUE.)
     END IF
     ! Make sure Q is full rank (for computing the rectangle component).
     FORALL (I = RANK+1 : K) Q(I,I) = 1.0
  END IF
  ! Store Vt (or Ut if A has more columns than rows).
  IF (PRESENT(VT)) VT(:,:) = Q(:,:)
  ! Compute U (or Vt if A has more columns than rows).
  IF (PRESENT(U)) THEN
     ! Compute the rectangle component.
     IF (SIZE(A,2) .LE. SIZE(A,1)) THEN
        PRINT *, 'N <= M (tall and skinny)'
        ! A = U s V
        ! U = (A Vt) / s
        ! U(:,:) = MATMUL(A(:,:), Q(:,:))
        CALL SGEMM('N', 'N', SIZE(U,1), SIZE(U,2), K, 1.0_RT, &
             A(:,:), SIZE(A,1), Q(:,:), SIZE(Q,1), 0.0_RT, &
             U(:,:), SIZE(U,1))
     ELSE
        PRINT *, 'N > M (short and wide)'
        ! At = Vt s Ut
        ! Vt = (At U) / s
        ! VT(:,:) = MATMUL(TRANSPOSE(A(:,:)), TRANSPOSE(Q(:,:)))
        CALL SGEMM('T', 'T', SIZE(U,1), SIZE(U,2), K, 1.0_RT, &
             A(:,:), SIZE(A,1), Q(:,:), K, 0.0_RT, &
             U(:,:), SIZE(U,1))
     END IF
     ! Apply the inverse of the singular values.
     DO I = 1, RANK
        IF (S(I) .NE. 0.0_RT) THEN
           U(:,I) = U(:,I) / S(I)
        ELSE
           U(:,I) = 0.0_RT
        END IF
     END DO
     ! Due to errors, U (Vt) might not be orthonormal. Fix that.
     PRINT *, 'Orthogonalizing the rectangular matrix (along length)..'
     CALL ORTHOGONALIZE(U(:,:), MAGS(:), RANK, TRANS=.TRUE., PIVOT=.FALSE.)
  END IF
  ! Deallocate.
  DEALLOCATE(AAT, Q, QTEMP)
END SUBROUTINE SVD


!2022-02-28 16:41:16
!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! ! ! Properly orthogonalize Q if U or VT is present.                 !
  ! ! IF ((PRESENT(VT) .OR. PRESENT(U)) .AND. (RANK .LT. K)) THEN       !
  ! !    ! Make the remainder of Q the identity.                        !
  ! !    Q(RANK+1:,:) = 0.0_RT                                          !
  ! !    FORALL (I = RANK+1:K) Q(I,I) = 1.0_RT                          !
  ! !    ! Rescale the column vectors to have the same original 2-norm. !
  ! !    DO I = 1, RANK                                                 !
  ! !       Q(:,I) = S(I) * Q(:,I) / SQRT(SUM(Q(:,I)**2))               !
  ! !    END DO                                                         !
  ! !    ! Redo orthogonalization knowing the rank is lower.            !
  ! !    CALL ORTHOGONALIZE(Q(:RANK,:RANK), S(:RANK))                   !
  ! ! END IF                                                            !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
